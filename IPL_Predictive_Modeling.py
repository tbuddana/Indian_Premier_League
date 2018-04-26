
# **Goal:** To build a predictive model to predict the winning probability of teams during the final match of IPL Season 2017

# Since the predictability of winning the final match in Season 2017 depends mainly on the compositions of the teams and their performances in Season 2017, we will consider only the Season 2017's data. 


from IPython import get_ipython
get_ipython().magic('reset -sf') 

import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_columns', 100)


# read the input files and look at the top few lines #
data_path = "/Users/venkatasravankanukolanu/Documents/Data Files/ipl/"
match= pd.read_csv(data_path+"matches.csv")
score= pd.read_csv(data_path+"deliveries.csv")


match_df=match[(match['season']==2017) & (match['dl_applied']==0)]
match_df.head()


# Since "id" from "match_df" and match_id from score_df are common columns, we can join "season" with score_df to subset for delivary level data from Season 2017

score_df=score.merge(match_df[['id','season','winner']],left_on='match_id',right_on='id', how='inner')
score_df.head()


# Now we need to think about the features that are required to build the model. To get started, will list the features in 2 different sets i.e. features that are important and features that are good to add. We can change them as we keep analyzing.

# 1. Runs scored in a given over
# 2. Wickets taken in a given over
# 3. Cumulative score by each over
# 4. Cumulative wickets taken by each over
# 5. Target that the team is chasing down
# 6. Remaining target by each over
# 7. Run rate
# 8. Required run rate
# 9. Difference between run rate and required run rate
# 10. Binary variables on whether the team for which we are predicting is batting team or bowling team
# 11. Total runs in the last 5 overs
# 12. Totals wickets in the last 5 overs


# Runs scored and wickets taken per over #
score_df.player_dismissed.fillna(0, inplace=True)
score_df['player_dismissed'].ix[score_df['player_dismissed'] != 0] = 1
train_df=score_df.groupby(['match_id','inning','over','batting_team','bowling_team','winner'])[['total_runs', 'player_dismissed']].agg(['sum']).reset_index()
train_df.columns=[['match_id','innings','over','batting_team','bowling_team','winner','runs_over','wkts_over']]
# Cumulative score and cumulative wickets taken by each over
train_df['innings_wickets'] = train_df.groupby(['match_id', 'innings'])['wkts_over'].cumsum()
train_df['innings_score'] = train_df.groupby(['match_id', 'innings'])['runs_over'].cumsum()


# Target that the team is chasing down. if first innings, target is -1#
score_inning1=train_df.groupby(['match_id','innings','batting_team','bowling_team'])['runs_over'].sum().reset_index()
score_inning1['innings']=np.where(score_inning1['innings']==1,2,1)
train_df=train_df.merge(score_inning1,how='left',on=['match_id', 'innings'])
train_df=train_df.drop(['batting_team_y', 'bowling_team_y'],axis=1)
train_df.columns=['match_id','innings','over','batting_team','bowling_team','winner','runs_over','wkts_over','innings_wkts','innings_runs','target']
first_innings_index = train_df[train_df.loc[:,'innings'] == 1].index
train_df.loc[first_innings_index, "target"] = -1
#train_df.head(40)


# Remaining target that the team is chasing down. if first innings, remaining target is -1#
train_df['remaining_target']=train_df['target']-train_df['innings_runs']
train_df.loc[first_innings_index, "remaining_target"] = -1
#train_df.head(40)

#Run rate
train_df['run_rate']=train_df['innings_runs']/train_df['over']
train_df.head()


# Required run rate. If first innings, required run rate is -1. If 20th over, equired run rate is 99 #
def get_required_rr(row):
    if row['remaining_target'] == -1:
        return -1.
    elif row['over'] == 20:
        return 99
    else:
        return row['remaining_target'] / (20-row['over'])
    
train_df['required_run_rate'] = train_df.apply(lambda row: get_required_rr(row), axis=1)

#Difference in run rate and required run rate. If first innings, it is -1#
def get_rr_diff(row):
    if row['innings'] == 1:
        return -1
    else:
        return row['run_rate'] - row['required_run_rate']
    
train_df['runrate_diff'] = train_df.apply(lambda row: get_rr_diff(row), axis=1)

#Response. If batting team is winner, set 1 elseif bowling_team is winner, set 0 #
#train_df['is_batting_team'] = (train_df['team1'] == train_df['batting_team']).astype('int')
train_df['is_batting_winner'] = (train_df['batting_team'] == train_df['winner']).astype('int')
train_df.head()


# ### Function to train a Xgboost model


x_cols = ['innings', 'over', 'runs_over', 'wkts_over', 'innings_wkts', 'innings_runs', 'target', 'remaining_target', 'run_rate', 'required_run_rate', 'runrate_diff']
# let us take all the matches but for the final as development sample and final as val sample #
val_df = train_df.ix[train_df.match_id == 59,:]
dev_df = train_df.ix[train_df.match_id != 59,:]

# create the input and target variables #
dev_X = np.array(dev_df[x_cols[:]])
dev_y = np.array(dev_df['is_batting_winner'])
val_X = np.array(val_df[x_cols[:]])[:-1,:]
val_y = np.array(val_df['is_batting_winner'])[:-1]
print(dev_X.shape, dev_y.shape)
print(val_X.shape, val_y.shape)


# define the function to create the model #
def runXGB(train_X, train_y, seed_val=0):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.05
    param['max_depth'] = 8
    param['silent'] = 1
    param['eval_metric'] = "auc"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = 100

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    model = xgb.train(plst, xgtrain, num_rounds)
    return model

# let us build the model and get predcition for the final match #
model = runXGB(dev_X, dev_y)
xgtest = xgb.DMatrix(val_X)
preds = model.predict(xgtest)

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i,feat))
    outfile.close()

create_feature_map(x_cols)
importance = model.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
imp_df = pd.DataFrame(importance, columns=['feature','fscore'])
imp_df['fscore'] = (imp_df['fscore'] / imp_df['fscore'].sum())*100

# create a function for labeling #
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                '%f' % float(height),
                ha='center', va='bottom')
        
labels = np.array(imp_df.feature.values)
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,6))
rects = ax.bar(ind, np.array(imp_df.fscore.values), width=width, color='y')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Importance score (%)")
ax.set_title("Variable importance")
autolabel(rects)
plt.show()
