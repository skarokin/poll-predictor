# Fetch data from college football API and save in CSV file
# in a separate script (create_dataset.py), we will split data by week and prepare data for training
import requests
import numpy as np
import os
from dotenv import load_dotenv

def max_weeks_in_year(games_data):
    max_week = 0
    for game in games_data:
        max_week = max(max_week, game['week'])
    return max_week

# optimization: pre-allocate numpy array w/ empty data 
# we have to directly calculate the number of weeks in the season
# to ensure we don't miss any games
def get_number_of_rows(START_YEAR, END_YEAR, API_BASE_URL, HEADERS):
    num_rows = 0
    for season in range(START_YEAR, END_YEAR + 1):
        games_response = requests.get(
            f"{API_BASE_URL}/games",
            headers=HEADERS,
            params={"year": season}
        )

        if games_response.status_code != 200:
            print(f"Failed to fetch data for season {season}: {games_response.status_code}")
            continue

        games_data = games_response.json()

        # our data is organized by team, so for every game a team plays in this season,
        # we need to add a new row
        teams = set()
        max_week = max_weeks_in_year(games_data)

        for game in games_data:
            teams.add(game['home_team'])
            teams.add(game['away_team'])

        season_rows = len(teams) * max_week
        num_rows += season_rows
    
    return int(num_rows * 1.1) # safety buffer just in case!

def get_fpi_stats(season, API_BASE_URL, HEADERS):
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
def get_polls(season, week, API_BASE_URL, HEADERS):
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
def get_game_results(season, week, API_BASE_URL, HEADERS):
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

# if unranked but moved to top 25, trend is (26 - new rank)
# if ranked but moved out of top 25, trend is (old rank - 26) 
def calculate_rank_trend(all_data):
    def rank_to_num(rank_str):
        try:
            return 26 if (rank_str == 'Unranked' or rank_str == '') else int(rank_str)
        except ValueError:
            return 26
        
    ap_trends = np.zeros(len(all_data), dtype=np.int32)
    coaches_trends = np.zeros(len(all_data), dtype=np.int32)

    current_team = None
    current_season = None
    prev_ap_rank = None
    prev_coaches_rank = None

    for i, row in enumerate(all_data):
        team = row['teamName']
        season = row['seasonYear']
        
        # Reset previous ranks when team or season changes
        if team != current_team or season != current_season:
            prev_ap_rank = None
            prev_coaches_rank = None
            current_team = team
            current_season = season
        
        # Calculate trends
        current_ap_rank = rank_to_num(row['apRank'])
        current_coaches_rank = rank_to_num(row['coachesRank'])
        
        if prev_ap_rank is not None:
            ap_trends[all_data[i]] = prev_ap_rank - current_ap_rank
        
        if prev_coaches_rank is not None:
            coaches_trends[all_data[i]] = prev_coaches_rank - current_coaches_rank
        
        # Update previous ranks
        prev_ap_rank = current_ap_rank
        prev_coaches_rank = current_coaches_rank
    
    return ap_trends, coaches_trends


def get_data(START_YEAR, END_YEAR, API_BASE_URL, HEADERS, all_data_dtype, all_data, current_index):
    # Populate data by season year and week
    for season in range(START_YEAR, END_YEAR + 1):
        print(f"Processing season {season}...")

        # Fetch FPI statistics for this season
        fpi_stats = get_fpi_stats(season, API_BASE_URL, HEADERS)

        # keep track of yearly team wins separately
        yearly_team_wins = np.array([], dtype=[('teamName', 'U100'), ('wins', 'i4'), ('losses', 'i4')])

        # Fetch game data for this season to calculate number of weeks for below loop
        games_response = requests.get(
            f"{API_BASE_URL}/games",
            headers=HEADERS,
            params={"year": season}
        )

        if games_response.status_code != 200:
            print(f"Failed to fetch data for season {season}: {games_response.status_code}")
            continue

        games_data = games_response.json()

        max_weeks = max_weeks_in_year(games_data)

        for week in range(1, max_weeks):
            print(f"  Processing week {week} of season {season}...")

            # Fetch AP and Coaches Poll rankings for the week
            # - dictionaries with team names as keys and rankings as values
            ap_rankings, coaches_rankings = get_polls(season, week, API_BASE_URL, HEADERS)

            # Fetch game results for the week
            # - a dictionary with team names as keys and a bunch of stats as values
            weekly_team_stats = get_game_results(season, week, API_BASE_URL, HEADERS)

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
                    week,
                    season,
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
                    0,  # placeholder for ap rank trend (it's easier to just calculate it at the end than in this loop)
                    0,  # placeholder for coaches rank trend
                )], dtype=all_data_dtype)
                
                # originally we were appending but that caused longer and longer times
                # to process as weeks progressed, this time it's the same speed regardless :)
                all_data[current_index] = new_row
                current_index += 1

    all_data = all_data[:current_index] # remove the safety buffer now since all data is populated

    # final step: calculate weekly trend for AP and Coaches Poll rankings
    trends = calculate_rank_trend(all_data)
    # no need for a mask here since calculate_rank_trend already takes care of matching 
    # team and season with their trend, just replace the columns in all_data
    print(trends)
    all_data['apRankTrend'] = trends[0]
    all_data['coachesRankTrend'] = trends[1]

    return all_data

def main():
    # Load environment variables
    load_dotenv()

    API_BASE_URL = os.environ.get("API_BASE_URL")
    API_KEY = os.environ.get("API_KEY")
    HEADERS = {"Authorization": f"Bearer {API_KEY}"}

    # Constants
    START_YEAR = 2005
    END_YEAR = 2005

    # Define dtype to match CSV format exactly
    all_data_dtype = np.dtype([
        ('teamName', 'U100'),
        ('weekNumber', 'i4'), 
        ('seasonYear', 'i4'),
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
        ('apRankTrend', 'i4'),
        ('coachesRankTrend', 'i4')
    ])

    num_rows = get_number_of_rows(START_YEAR, END_YEAR, API_BASE_URL, HEADERS)
    print(f"Total number of rows: {num_rows}")
    all_data = np.zeros(num_rows, dtype=all_data_dtype)
    current_index = 0

    all_data = get_data(START_YEAR, END_YEAR, API_BASE_URL, HEADERS, all_data_dtype, all_data, current_index)

    # Save data as csv
    # note that model expects week-splitted data, but to ensure ease of use, we save as a CSV and post-process it for model input
    np.savetxt(
        "college_football_stats_with_detailed_fpi_and_ap_2005_to_2023_weekly_numpy.csv", 
        all_data, # don't take the safety buffer 
        delimiter=',', 
        fmt='%s', 
        header=','.join(all_data_dtype.names), 
        comments=''
    )

    print("Data export complete!")

main()