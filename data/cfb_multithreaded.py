# this would've been amazing BUT the api is rate limited so

import requests
import numpy as np
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Load environment variables
load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
WAIT_TIME = 0.8

# Constants
start_year = 2005
end_year = 2023
current_week = 13  

all_data = []

all_data_dtype = np.dtype([
    ('teamName', 'U100'), ('record', 'U10'), ('fpi', 'f8'), ('strengthOfRecord', 'f8'), 
    ('averageWinProbability', 'f8'), ('strengthOfSchedule', 'f8'), ('gameControl', 'f8'), 
    ('overallEfficiency', 'f8'), ('offenseEfficiency', 'f8'), ('defenseEfficiency', 'f8'), 
    ('specialTeamsEfficiency', 'f8'), ('apRank', 'i4'), ('coachesRank', 'i4'), 
    ('weekNumber', 'i4'), ('seasonYear', 'i4')
])

def get_fpi_stats(season): 
    time.sleep(WAIT_TIME)
    response_fpi = requests.get(
        f"{API_BASE_URL}/ratings/fpi",
        headers=HEADERS,
        params={"year": season}
    )

    if response_fpi.status_code != 200:
        print(f"Failed to fetch FPI data for {season}: {response_fpi.status_code}")
        return np.array([])
    
    fpi_data = response_fpi.json()

    # use structured and pre-allocated numpy arrays for performance over dictionaries
    # since originally we had team names as keys, we need to have this identifier in the array
    team_count = len(fpi_data)
    dtype = [
        ('teamName', 'U100'),
        ('fpi', 'f8'),
        ('strengthOfRecord', 'f8'),
        ('averageWinProbability', 'f8'),
        ('strengthOfSchedule', 'f8'),
        ('gameControl', 'f8'),
        ('overall_efficiency', 'f8'),
        ('offense_efficiency', 'f8'),
        ('defense_efficiency', 'f8'),
        ('specialTeams_efficiency', 'f8')
    ]

    fpi_stats = np.zeros(team_count, dtype=dtype)

    for i, team in enumerate(fpi_data):
        efficiencies = team.get("efficiencies", {})
        fpi_stats[i] = (
            team["team"],
            team["fpi"],
            team["resumeRanks"]["strengthOfRecord"],
            team["resumeRanks"]["averageWinProbability"],
            team["resumeRanks"]["strengthOfSchedule"],
            team["resumeRanks"]["gameControl"],
            efficiencies.get("overall", np.nan),
            efficiencies.get("offense", np.nan),
            efficiencies.get("defense", np.nan),
            efficiencies.get("specialTeams", np.nan)
        )
    
    return fpi_stats

# we are only interested in AP and Coaches' Polls
def get_polls(season, week):
    time.sleep(WAIT_TIME)
    response_polls = requests.get(
        f"{API_BASE_URL}/rankings",
        headers=HEADERS,
        params={"year": season, "week": week}
    )

    if response_polls.status_code != 200:
        print(f"Failed to fetch AP rankings for week {week} of season {season}: {response_polls.status_code}")
        return np.array([]), np.array([])

    rankings_data = response_polls.json()

    # again, use structured and pre-allocated numpy arrays for performance
    dtype = [('teamName', 'U100'), ('rank', 'i4')]
    ap_rankings = []
    coaches_rankings = []

    for ranking in rankings_data:
        for poll in ranking.get("polls", []):
            if poll["poll"] == "AP Top 25":
                for rank_entry in poll["ranks"]:
                    school = rank_entry.get("school") # equivalent to teamName
                    rank = rank_entry.get("rank")
                    ap_rankings.extend([(school, rank)])
            elif poll["poll"] == "Coaches Poll":
                for rank_entry in poll["ranks"]:
                    school = rank_entry.get("school") # equivalent to teamName
                    rank = rank_entry.get("rank")
                    coaches_rankings.extend([(school, rank)])

    return np.array(ap_rankings, dtype=dtype), np.array(coaches_rankings, dtype=dtype)

# instead of updating yearly_team_stats dictionary, we are now creating a new numpy array
# so, whatever is returned is used to update yearly_team_stats
def get_game_results(season, week):
    time.sleep(WAIT_TIME)
    response_games = requests.get(
        f"{API_BASE_URL}/games",
        headers=HEADERS,
        params={"year": season, "week": week}
    )

    if response_games.status_code != 200:
        print(f"Failed to fetch game data for week {week} of season {season}: {response_games.status_code}")
        return np.array([])
    
    games = response_games.json()

    dtype = [('teamName', 'U100'), ('wins', 'i4'), ('losses', 'i4')]
    game_results = []

    for game in games:
        home_points = game.get("home_points", 0) or 0
        away_points = game.get("away_points", 0) or 0

        home_team = game["home_team"]
        home_win = 1 if home_points > away_points else 0
        home_loss = 1 if home_points < away_points else 0
        game_results.append((home_team, home_win, home_loss))

        away_team = game["away_team"]
        away_win = 1 if away_points > home_points else 0
        away_loss = 1 if away_points < home_points else 0
        game_results.append((away_team, away_win, away_loss))
    
    return np.array(game_results, dtype=dtype)

def process_week(season, week, yearly_team_wins, fpi_stats):
    ap_rankings, coaches_rankings = get_polls(season, week)
    weekly_team_stats = get_game_results(season, week)

    for result in weekly_team_stats:
        team_mask = yearly_team_wins['teamName'] == result['teamName']
        if any(team_mask):
            yearly_team_wins[team_mask]['wins'] += result['wins']
            yearly_team_wins[team_mask]['losses'] += result['losses']
        else:
            yearly_team_wins = np.append(yearly_team_wins, result)

    weekly_data = []
    for team in yearly_team_wins:
        record = f"{team['wins']}-{team['losses']}"
        team_name = team['teamName']

        fpi_mask = fpi_stats['teamName'] == team_name
        fpi_detail = fpi_stats[fpi_mask][0] if any(fpi_mask) else None

        ap_mask = ap_rankings['teamName'] == team_name
        coaches_mask = coaches_rankings['teamName'] == team_name

        ap_rank = ap_rankings[ap_mask]['rank'][0] if any(ap_mask) else "Unranked"
        coaches_rank = coaches_rankings[coaches_mask]['rank'][0] if any(coaches_mask) else "Unranked"

        weekly_data.append((
            team_name, record,
            fpi_detail['fpi'] if fpi_detail else np.nan,
            fpi_detail['strengthOfRecord'] if fpi_detail else np.nan,
            fpi_detail['averageWinProbability'] if fpi_detail else np.nan,
            fpi_detail['strengthOfSchedule'] if fpi_detail else np.nan,
            fpi_detail['gameControl'] if fpi_detail else np.nan,
            fpi_detail['overall_efficiency'] if fpi_detail else np.nan,
            fpi_detail['offense_efficiency'] if fpi_detail else np.nan,
            fpi_detail['defense_efficiency'] if fpi_detail else np.nan,
            fpi_detail['specialTeams_efficiency'] if fpi_detail else np.nan,
            ap_rank, coaches_rank, week, season
        ))

    return weekly_data

def process_year(season):
    yearly_team_wins = np.array([], dtype=[('teamName', 'U100'), ('wins', 'i4'), ('losses', 'i4')])
    fpi_stats = get_fpi_stats(season)

    yearly_data = []
    with ThreadPoolExecutor() as week_executor:
        week_futures = [
            week_executor.submit(process_week, season, week, yearly_team_wins.copy(), fpi_stats)
            for week in range(1, current_week + 1)
        ]

        for future in as_completed(week_futures):
            yearly_data.extend(future.result())

    return yearly_data

# Multithreading for years
with ThreadPoolExecutor() as year_executor:
    year_futures = [year_executor.submit(process_year, season) for season in range(start_year, end_year + 1)]

    for future in as_completed(year_futures):
        all_data.extend(future.result())

# Sort data by year and week since multithreading by year AND by week can cause out-of-order data
all_data = np.array(all_data, dtype=all_data_dtype)
all_data.sort(order=['seasonYear', 'weekNumber'])

# Save sorted data as CSV
np.savetxt(
    "college_football_stats_sorted.csv", 
    all_data, 
    delimiter=',', 
    fmt='%s', 
    header=','.join(all_data_dtype.names), 
    comments=''
)

print("Data export complete!")
