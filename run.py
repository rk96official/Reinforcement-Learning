
from deepLearning.agents.agent import Agent
from deepLearning.nlu.nlu import set_nlu
from score.score_model import *
from episodes import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

model_path=None
params = parameters()

df, dict_intents, top_intents = load_data(params, max_intents = 6)

NLU = set_nlu(params, 'rasa', top_intents)

mapping = get_mapping(NLU, df) #defined in utils.py

simulator = Simulator(df, multiply=4)

warm_start = 1
agent = Agent(NLU, params, mapping, warm_start, model_path=None)

if model_path is None:
        
        agent, Q_table = warmup_run(agent, simulator, params, mapping, use_Q=False, verbose=False)

        # test DQN accuracy by cfr DQN with NLU results
        testing(agent, mapping, Q_table, verbose=False)
        print('... END of WARM-UP PHASE')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


#######################################################
#  Train score model
######################################################

# load score mdoel configuration (in utils.py)
params_model = config_score_model()
    
# load dat to train score model (utils.py)
x, y, emb_u, emb_r = score_data()

#init score model (in learning_scores/score_model.py)
model = score_model(params_model)

print('Training score model...')
Final_train_acc, _ = model.fit(x,y, emb_u=emb_u, emb_r=emb_r, path_to_model = "./models/model_final.ckpt")
    
print('Train Accuracy of score model:', Final_train_acc)
print('----------------------------------------------------')


#######################################################
#  Run episodes
######################################################
print('---------- Running RL episodes -----------')
results = episodes_run(agent, NLU, simulator, model,  params, mapping)

print('End of episodes !')
print()          

# write summary of results to file
results.to_csv('summary.csv', index=False)






