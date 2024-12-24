# Fetch data from college football API and save in CSV file
# in a separate script (create_dataset.py), we will split data by week and prepare data for training
import requests
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Constants
start_year = 2005
end_year = 2023 
current_week = 13  

# Define dtype to match CSV format exactly
all_data_dtype = np.dtype([
    ('teamName', 'U100'),
    ('record', 'U10'),
    ('fpi', 'U20'),  
    ('strengthOfRecord', 'U20'),
    ('averageWinProbability', 'U20'),
    ('strengthOfSchedule', 'U20'),
    ('gameControl', 'U20'),
    ('overallEfficiency', 'U20'),  
    ('offenseEfficiency', 'U20'),  
    ('defenseEfficiency', 'U20'),  
    ('specialTeamsEfficiency', 'U20'), 
    ('apRank', 'U20'),  
    ('coachesRank', 'U20'), 
    ('weekNumber', 'i4'), 
    ('seasonYear', 'i4')  
])

# optimization: pre-allocate numpy array w/ empty data 
# num_rows = (seasons) * (weeks) * (teams) = (2023-2005+1) * 13 * 130
# - a safer way to calculate teams is to fetch number of teams from API but... nah too much work
# then, we can just update indexes instead of expensive appends
all_data = np.array([], dtype=all_data_dtype)

def get_fpi_stats(season): 
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
        # original data had team as key, but since we are using numpy arrays, we need to add the key as a field for identification later
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
    response_polls = requests.get(
        f"{API_BASE_URL}/rankings",
        headers=HEADERS,
        params={"year": season, "week": week}
    )

    if response_polls.status_code != 200:
        print(f"Failed to fetch rankings for week {week} of season {season}: {response_polls.status_code}")
        return np.array([]), np.array([])

    rankings_data = response_polls.json()

    # again, use structured and pre-allocated numpy arrays for performance
    dtype = [('teamName', 'U100'), ('rank', 'i4')]
    ap_rankings = []
    coaches_rankings = []

    for ranking in rankings_data:
        for poll in ranking.get("polls", []):
            if poll["poll"] == "AP Top 25":
                ap_rank = None
                coaches_rank = None
                for rank_entry in poll["ranks"]:
                    school = rank_entry.get("school") # equivalent to teamName
                    ap_rank = rank_entry.get("rank")
                    ap_rankings.extend([(school, ap_rank)])
            elif poll["poll"] == "Coaches Poll":
                for rank_entry in poll["ranks"]:
                    school = rank_entry.get("school") # equivalent to teamName
                    coaches_rank = rank_entry.get("rank")
                    coaches_rankings.extend([(school, coaches_rank)])

    return np.array(ap_rankings, dtype=dtype), np.array(coaches_rankings, dtype=dtype)

# instead of updating yearly_team_stats dictionary, we are now creating a new numpy array
# so, whatever is returned is used to update yearly_team_stats
def get_game_results(season, week):
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
        for team, _, points, opp_points in [
                        (game["home_team"], game["away_team"], game.get("home_points"), game.get("away_points")),
                        (game["away_team"], game["home_team"], game.get("away_points"), game.get("home_points")),
                    ]:
                points = points or 0
                opp_points = opp_points or 0

                if points > opp_points:
                    game_results.append((team, 1, 0))
                elif points < opp_points:
                    game_results.append((team, 0, 1))
                # if tie, don't update
    
    return np.array(game_results, dtype=dtype)

# Populate data by season year and week
for season in range(start_year, end_year + 1):
    print(f"Processing season {season}...")

    # Fetch FPI statistics for this season
    fpi_stats = get_fpi_stats(season)

    # keep track of yearly team wins separately
    yearly_team_wins = np.array([], dtype=[('teamName', 'U100'), ('wins', 'i4'), ('losses', 'i4')])

    for week in range(1, current_week + 1):
        print(f"  Processing week {week} of season {season}...")

        # Fetch AP and Coaches Poll rankings for the week
        # - dictionaries with team names as keys and rankings as values
        ap_rankings, coaches_rankings = get_polls(season, week)

        # TODO: Calculate weekly trend for AP and Coaches Poll rankings

        # Fetch game results for the week
        # - a dictionary with team names as keys and a bunch of stats as values
        weekly_team_stats = get_game_results(season, week)

        # Update yearly_team_wins with the weekly results
        for result in weekly_team_stats:
            team_mask = yearly_team_wins['teamName'] == result['teamName']
            if any(team_mask):
                # Update wins and losses for the existing team
                yearly_team_wins['wins'][team_mask] += result['wins']
                yearly_team_wins['losses'][team_mask] += result['losses']
            else:
                # Add new team to yearly_team_wins
                new_entry = np.array([(result['teamName'], result['wins'], result['losses'])], dtype=yearly_team_wins.dtype)
                yearly_team_wins = np.append(yearly_team_wins, new_entry)

        for team in yearly_team_wins:
            record = f"{team['wins']}-{team['losses']}"
            team_name = team['teamName']

            fpi_mask = fpi_stats['teamName'] == team_name
            fpi_detail = fpi_stats[fpi_mask][0] if any(fpi_mask) else None

            ap_mask = ap_rankings['teamName'] == team_name
            coaches_mask = coaches_rankings['teamName'] == team_name
            ap_rank = ap_rankings[ap_mask]['rank'][0] if any(ap_mask) else "Unranked"
            coaches_rank = coaches_rankings[coaches_mask]['rank'][0] if any(coaches_mask) else "Unranked"

            

            new_row = np.array([(
                team_name,
                record, 
                str(fpi_detail['fpi'] if fpi_detail else "N/A"),
                str(fpi_detail['strengthOfRecord'] if fpi_detail else "N/A"),
                str(fpi_detail['averageWinProbability'] if fpi_detail else "N/A"),
                str(fpi_detail['strengthOfSchedule'] if fpi_detail else "N/A"),
                str(fpi_detail['gameControl'] if fpi_detail else "N/A"),
                str(fpi_detail['overall_efficiency'] if fpi_detail else "N/A"),
                str(fpi_detail['offense_efficiency'] if fpi_detail else "N/A"),
                str(fpi_detail['defense_efficiency'] if fpi_detail else "N/A"),
                str(fpi_detail['specialTeams_efficiency'] if fpi_detail else "N/A"),
                str(ap_rank),
                str(coaches_rank),
                week,
                season
            )], dtype=all_data_dtype)
            
            all_data = np.concatenate((all_data, new_row))

# Save data as csv
# note that model expects week-splitted data, but to ensure ease of use, we save as a CSV and post-process it for model input
np.savetxt(
    "college_football_stats_with_detailed_fpi_and_ap_2005_to_2023_weekly_numpy.csv", 
    all_data, 
    delimiter=',', 
    fmt='%s', 
    header=','.join(all_data_dtype.names), 
    comments=''
)

print("Data export complete!")