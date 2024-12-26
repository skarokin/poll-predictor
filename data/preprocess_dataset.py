import pandas as pd 
import numpy as np

# ground truths are next week's apRank for each team at each week
# note that coaches rank is our simulated community poll, and AP is the ground truth we chose
def create_ground_truth(data):
    # create a key for current week
    data['currentWeekId'] = data['teamName'] + '_' + data['seasonYear'].astype(str) + '_' + (data['weekNumber']).astype(str)
    
    # create a key for next week
    data['nextWeekId'] = data['teamName'] + '_' + data['seasonYear'].astype(str) + '_' + (data['weekNumber'] + 1).astype(str)

    # this just creates a dictionary that maps currentWeekId to apRank
    # the nature of this means next week's game can simply be looked up with the nextWeekId
    all_week_ranks = data.set_index('currentWeekId')['apRank'].to_dict()

    # since all_week_ranks maps ALL weeks to their apRank, we can then
    # populate nextWeekRank by simply looking up team X's next week game (this is the purpose of unique identifier for
    # next week's game) and looking up the apRank for that week
    # even if the next week's game is not in the data (for example end of season), the nextWeekRank will be NaN
    data['nextWeekRank'] = data['nextWeekId'].map(all_week_ranks)

    # create ground truth
    ground_truth = data[['currentWeekId', 'nextWeekRank']]

    return data, ground_truth

# transform raw data into trainable ata
# 1. transform 'record' to 'winPercentage' (wins / total games)
# 2. add a 'winDominance' column (calculate home - away points... if team is away, then -1 * winDominance)
# 3. any stat that didn't have to be calculated not scaled to [0, 1] should be divided or multiplied to get it to [-1, 1]
#    this means any boolean columns should be 0 or 1
# 4. replace 'Unranked' with 26
#   4a. add a new column 'wasRankedLastWeek' that is simply 1 if last week's rank was not 26, 0 otherwise
# 5. scale weekly rank trends from -1 to 1 
#      - max possible rank is 26 (unranked), min is 1, meaning trends range from [-25, 25]
#      - so, divide by 25 to get [-1, 1]
# 6. replace all N/A with 0 (this is okay because teams without data are known to be subpar anyways)
# 7. give every row the opponent's data as well (easy with unique identifier)
def preprocess(data):
    # 1 - record to winPercentage
    def record_to_win_percentage(record):
        wins, losses = record.split('-')
        return int(wins) / (int(wins) + int(losses))
    data['record'] = data['record'].apply(record_to_win_percentage)
    data = data.rename(columns={'record': 'winPercentage'})

    # 2 - add winDominance
    win_dominance = data['homePoints'] - data['awayPoints']
    data['winDominance'] = np.where(data['isHome'] == 'True', win_dominance, -1 * win_dominance)

    # 3 - scale stats
    # fpi, strength of record, average win probability, game control, {overall, offense, defense, special teams} efficiency are 100-scaled
    # strength of schedule not on a scale at all, so we need to find min and max then scale
    min_sos = data['strengthOfSchedule'].min()
    max_sos = data['strengthOfSchedule'].max()
    data['strengthOfSchedule'] = data['strengthOfSchedule'].apply(lambda x: (x - min_sos) / (max_sos - min_sos))
    data[['fpi', 'strengthOfRecord', 'averageWinProbability', 'gameControl', 'overallEfficiency',
         'offenseEfficiency', 'defenseEfficiency', 'specialTeamsEfficiency']] = \
          data[['fpi', 'strengthOfRecord', 'averageWinProbability', 'gameControl', 'overallEfficiency',
               'offenseEfficiency', 'defenseEfficiency', 'specialTeamsEfficiency']].apply(lambda x: x / 100)
    
    bool_cols = ['isHome', 'didWin']
    data[bool_cols] = data[bool_cols].replace({'True': 1, 'False': 0}).astype(int)
    
    # 4 - replace 'Unranked' with 26
    data.replace('Unranked', 26, inplace=True)

    #   4a - add wasRankedLastWeek
    all_week_ranks = data.set_index('currentWeekId')['apRank'].to_dict()
    data['lastWeekId'] = data['teamName'] + '_' + data['seasonYear'].astype(str) + '_' + (data['weekNumber'] - 1).astype(str)
    data['wasRankedLastWeek'] = data.apply(
        lambda x:
            0 if
                x['weekNumber'] == 1
                or all_week_ranks.get(x['lastWeekId']) == 26
                or all_week_ranks.get(x['lastWeekId']) is None
            else 1, 
        axis=1)

    # 5 - scale weekly rank trends
    data['apRankTrend'] = data['apRankTrend'].apply(lambda x: x / 25)
    data['coachesRankTrend'] = data['coachesRankTrend'].apply(lambda x: x / 25)

    # 6 - replace N/A with 0
    # unfortunately i tried in the data gathering script to replace None with N/A for postWinProbability but it didnt work, anyways
    # im too lazy and this is not a big deal at all
    data.replace('None', np.nan, inplace=True)
    data.fillna(0, inplace=True)

    # 7 - add opponent data (what I mean is the opponent from this game, not last week's opponent or next week's projected opponent)
    # this allows the model to better understand how ranking is affected by both teams' performance, not just the one we are looking at 
    
    # build a series of opponent week ids
    opponent_week_ids = data['opponentTeamName'] + '_' + data['seasonYear'].astype(str) + '_' + data['weekNumber'].astype(str)
    # change index to currentWeekId; use a copy to ensure original index is preserved
    opponent_data = data.copy()
    opponent_data.set_index('currentWeekId', inplace=True)
    # get opponent data
    opponent_data = opponent_data.loc[opponent_week_ids]
    # reset index to ensure proper merge and column rename/dropping
    opponent_data.reset_index(inplace=True) # don't drop (we still have to use it lol)
    # drop unnecessary columns and rename
    opponent_data = opponent_data.drop(columns=['weekNumber', 'seasonYear', 'opponentTeamName', 
                                          'currentWeekId', 'nextWeekId', 'lastWeekId'])
    opponent_data.columns = ['opponent_' + col for col in opponent_data.columns]
    # merge
    data = data.join(opponent_data) # works because order is preserved

    return data

def main():
    # read csv 
    csv_path = 'cfb_2005_2024.csv'
    data = pd.read_csv(csv_path)
    data.sort_values(by=['seasonYear', 'weekNumber'], inplace=True)

    data, ground_truth = create_ground_truth(data)
    print('ground truth created')

    training_ready = preprocess(data)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    print('data preprocessed\n', training_ready)
    
if __name__ == '__main__':
    main()