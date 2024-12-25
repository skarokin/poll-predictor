import pandas as pd 
import numpy as np

# ground truths are next week's apRank for each team at each week
def create_ground_truth():
    # read csv 
    csv_path = 'cfb_2005_2024.csv'
    data = pd.read_csv(csv_path)
    data.sort_values(by=['seasonYear', 'weekNumber'], inplace=True)

    # add unique identifier column (for use in creating ground truth only), not used in calculating nextWeekRank
    data['uniqueId'] = data['seasonYear'].astype(str) + '_' + data['weekNumber'].astype(str) + '_' + data['teamName']

    # create a key for current week
    data['currentWeekKey'] = data['teamName'] + '_' + data['seasonYear'].astype(str) + '_' + data['weekNumber'].astype(str)
    
    # create a key for next week
    data['nextWeekKey'] = data['teamName'] + '_' + data['seasonYear'].astype(str) + '_' + (data['weekNumber'] + 1).astype(str)

    # map each currentWeekKey to its corresponding apRank
    all_week_ranks = data.set_index('currentWeekKey')['apRank'].to_dict()

    # since all_week_ranks maps ALL weeks to their apRank, we can then
    # populate nextWeekRank by simply looking up team X's next week game (this is the purpose of unique identifier for
    # next week's game) and looking up the apRank for that week
    # even if the next week's game is not in the data (for example end of season), the nextWeekRank will be NaN
    data['nextWeekRank'] = data['nextWeekKey'].map(all_week_ranks)

    # create ground truth
    ground_truth = data[['uniqueId', 'nextWeekRank']]

    return ground_truth

# TODO: convert string features to numerical features (for attention mechanism)
def quantify_stats():
    pass

def main():
    ground_truth = create_ground_truth()

if __name__ == '__main__':
    main()