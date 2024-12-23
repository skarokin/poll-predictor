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

all_data = []

columns = [
    "Team Name", "W/L till today", "FPI", "Strength of Record", "Average Win Probability", 
    "Strength of Schedule", "Game Control", 
    "Overall Efficiency", "Offense Efficiency", "Defense Efficiency", "Special Teams Efficiency", 
    "AP Poll Ranking", "Coaches Poll", "Week Number", "Season Year"
]

def get_fpi_stats(season):
    fpi_stats = {}
    
    response_fpi = requests.get(
        f"{API_BASE_URL}/ratings/fpi",
        headers=HEADERS,
        params={"year": season}
    )

    if response_fpi.status_code == 200:
        fpi_data = response_fpi.json()
        for team in fpi_data:
            efficiencies = team.get("efficiencies", {})
            fpi_stats[team["team"]] = {
                "fpi": team["fpi"],
                "strengthOfRecord": team["resumeRanks"]["strengthOfRecord"],
                "averageWinProbability": team["resumeRanks"]["averageWinProbability"],
                "strengthOfSchedule": team["resumeRanks"]["strengthOfSchedule"],
                "gameControl": team["resumeRanks"]["gameControl"],
                "efficiencies": efficiencies
            }
    else:
        print(f"Failed to fetch FPI data for {season}: {response_fpi.status_code}")
    
    return fpi_stats

# we are only interested in AP and Coaches' Polls
def get_polls(season, week):
    ap_rankings = {}
    coaches_rankings = {}
    
    response_polls = requests.get(
        f"{API_BASE_URL}/rankings",
        headers=HEADERS,
        params={"year": season, "week": week}
    )

    if response_polls.status_code == 200:
        rankings_data = response_polls.json()
        for ranking in rankings_data:
            polls = ranking.get("polls", [])
            for poll in polls:
                if poll["poll"] == "AP Top 25":
                    for rank_entry in poll["ranks"]:
                        school = rank_entry.get("school")
                        rank = rank_entry.get("rank")
                        if school and rank is not None:
                            ap_rankings[school] = rank
                elif poll["poll"] == "Coaches Poll":
                    for rank_entry in poll["ranks"]:
                        school = rank_entry.get("school")
                        rank = rank_entry.get("rank")
                        if school and rank is not None:
                            coaches_rankings[school] = rank
    else:
        print(f"Failed to fetch AP rankings for week {week} of season {season}: {response_polls.status_code}")
    
    return ap_rankings, coaches_rankings

# need to pass in current_team_stats to ensure we are updating yearly based on weekly data, not resetting to 0-0 every week
def get_game_results(season, week, current_team_stats):
    
    response_games = requests.get(
        f"{API_BASE_URL}/games",
        headers=HEADERS,
        params={"year": season, "week": week}
    )

    if response_games.status_code == 200:
        games = response_games.json()
        for game in games:
            for team, _, points, opp_points in [
                (game["home_team"], game["away_team"], game.get("home_points"), game.get("away_points")),
                (game["away_team"], game["home_team"], game.get("away_points"), game.get("home_points")),
            ]:
                points = points or 0
                opp_points = opp_points or 0
                if team not in current_team_stats:
                    current_team_stats[team] = {"wins": 0, "losses": 0}
                if points > opp_points:
                    current_team_stats[team]["wins"] += 1
                elif points < opp_points:
                    current_team_stats[team]["losses"] += 1
    else:
        print(f"Failed to fetch game data for week {week} of season {season}: {response_games.status_code}")
    
    return current_team_stats

# Populate data by season year and week
for season in range(start_year, end_year + 1):
    print(f"Processing season {season}...")

    # Fetch FPI statistics for this season
    fpi_stats = get_fpi_stats(season)

    # keep track of team stats for the season to be updated weekly and reset yearly
    yearly_team_stats = {}

    for week in range(1, current_week + 1):
        print(f"  Processing week {week} of season {season}...")

        # Fetch AP and Coaches Poll rankings for the week
        # - dictionaries with team names as keys and rankings as values
        ap_rankings, coaches_rankings = get_polls(season, week)

        # Fetch game results for the week
        # - a dictionary with team names as keys and a bunch of stats as values
        team_stats = get_game_results(season, week, yearly_team_stats)

        # the nature of appending maintains the chronological order of data
        # this is important for the model to understand the progression of the season
        for team, stats in team_stats.items():
            record = f"{stats['wins']}-{stats['losses']}"

            fpi_detail = fpi_stats.get(team, {})
            fpi = fpi_detail.get("fpi", "N/A")
            strengthOfRecord = fpi_detail.get("strengthOfRecord", "N/A")
            averageWinProbability = fpi_detail.get("averageWinProbability", "N/A")
            strengthOfSchedule = fpi_detail.get("strengthOfSchedule", "N/A")
            gameControl = fpi_detail.get("gameControl", "N/A")
            efficiencies = fpi_detail.get("efficiencies", {})
            overall_efficiency = efficiencies.get("overall", "N/A")
            offense_efficiency = efficiencies.get("offense", "N/A")
            defense_efficiency = efficiencies.get("defense", "N/A")
            specialTeams_efficiency = efficiencies.get("specialTeams", "N/A")
            
            ap_rank = ap_rankings.get(team, "Unranked")
            coaches_rank = coaches_rankings.get(team, "unranked")
            
            all_data.append([
                team, record, fpi, strengthOfRecord, averageWinProbability, 
                strengthOfSchedule, gameControl, 
                overall_efficiency, offense_efficiency, defense_efficiency, 
                specialTeams_efficiency, ap_rank, coaches_rank, week, season
            ])

# Save data as csv
# note that model expects week-splitted data, but to ensure ease of use, we save as a CSV and post-process it for model input
all_data_array = np.array(all_data)
np.savetxt(
    "college_football_stats_with_detailed_fpi_and_ap_2005_to_2023_weekly.csv", 
    all_data_array, 
    delimiter=',', 
    fmt='%s', 
    header=','.join(columns), 
    comments=''
)

print("Data export complete!")