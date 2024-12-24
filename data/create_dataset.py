# 1. load CSV data
# 2. split data by week
# 3. add unique game identifiers
# 4. create ground truth NP array, which is (game_id, next_week_rank)
# its just like 'next week ranking = the row in all_data where team name matches, week = week + 1, and season = season'
# and if week + 1 mask is empty, next week rating = N/A