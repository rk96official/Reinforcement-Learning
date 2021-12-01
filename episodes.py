from utils import *
from deepLearning.nlu.q_table import Q_table as Q_tab
from evaluate import *
from statsmodels.stats.proportion import proportion_confint
import os
import copy
import pandas as pd
import pickle


def episodes_run(agent, NLU, simulator, model, params, mapping, output='epochs.csv', use_model = True, thre_no_intent=0.5):

    ### Configure RL agent
    num_episodes = params['num_episodes']
    train_freq = params['train_freq']

    map_state2index = mapping['state2index']
    map_index2state = mapping['index2state']
    map_index2action = mapping['index2action']
    map_action2index = mapping['action2index']

    map_action2answer = pd.read_csv('responses.csv')

    emb_states = all_embeddings(map_index2state)
    emb_actions  = all_embeddings(map_index2action,  map_action = map_action2answer)

    eval = evaluate(map_state2index, map_index2action)
        
    sum_rewards, sum_rewards_bin, count_episodes, epoch = 0, 0, 0, 0
    summary = []

    flush = False

    utterance = simulator.run_random()

    for episode in range(num_episodes):

        s_t = utterance
        print('----------------------------------------')
        print('Episode: %d' % episode)
        print('utterance: %s' % utterance)

        repr = get_representation(utterance, map_state2index)

        index_action = agent.run_policy(repr, epoch)

        action = map_index2action[index_action]

        print('intent:', action)

        if not use_model: #interactive rewards
            reward_int = -99
            reward_int = get_reward(utterance, action)
            reward = float(reward_int)
        else: # reward from score model
            emb_u = emb_states[map_state2index[utterance]]
            emb_a = emb_actions[index_action]
            reward_model = get_reward_model([utterance], [action], [emb_u], [emb_a], model, map_action2answer)
            reward = reward_model

            if action == 'No intent detected':
                other_utt = [utterance]*(len(map_action2index)-1)
                other_act = list(map_action2index.keys())
                other_act.remove('No intent detected')         
                emb_u = [emb_states[map_state2index[utterance]]] *(len(map_action2index)-1)
                emb_a = [emb_actions[map_action2index[action]] for action in other_act]                                            
                rs = get_reward_model(other_utt, other_act, emb_u, emb_a, model, map_action2answer)
 
                if np.array([rs >= thre_no_intent]).any():
                    reward = 1.
                    
        print('Reward: %d' % reward)
        reward_bin = 0 if reward<0.5 else 1
        s_t_plus1 = simulator.run_random()
        episode_over = True
        sum_rewards = sum_rewards + float(reward)
        sum_rewards_bin = sum_rewards_bin + reward_bin
        count_episodes = count_episodes + 1
        fill_buffer_all_actions(agent, s_t, action, reward, s_t_plus1, episode_over, flush=flush)

        if (episode > 0 and (episode+1) % train_freq == 0) :

            print('TRAINING DQN agent...')
            agent.clone_dqn = copy.deepcopy(agent.dqn)
            agent.train(4, 10, num_iter = 100)
            success_rate = eval.fit(agent) #success rate on test set
            avg_score = float(sum_rewards)/float(count_episodes)
            avg_score_bin = float(sum_rewards_bin)/float(count_episodes)
            print('Average score for this epoch: %f, %f'%(avg_score, avg_score_bin))
            print('Success_rate on test set: %f' % success_rate)
            summary.append({'epoch': epoch, 'avg_score':avg_score, 'avg_score_bin': avg_score_bin, 'success_rate':success_rate})
            print('Summary:', summary)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            save_model('./models/', agent, epoch)
            
            if avg_score > 0.7:
                flush = True
                
            sum_rewards = 0
            sum_rewards_bin = 0
            count_episodes = 0
            epoch += 1

        utterance = s_t_plus1
        df_out = pd.DataFrame(summary, columns = ['epoch', 'avg_score', 'avg_score_bin', 'success_rate'] )
    return df_out 


def get_reward_model(utterances, actions, emb_u, emb_a, model, map_action2answer):
    embed_size = np.size(emb_u[0])
    m = len(utterances)
    x = np.zeros([2,m,embed_size])

    for i in range(m):
        x[0,i,:] = np.array(emb_u[i])
        x[1,i,:] = np.array(emb_a[i])

    pred = model.predict(x, path_to_model = "./score/trained_model/model_final.ckpt")
    return pred


def get_representation(utterance, map_state2index):
    size = len(map_state2index)
    index = map_state2index[utterance]
    rep = np.zeros((1,size))
    rep[0, index] = 1.0
    return rep


def all_embeddings(map_index2sent, map_action=None):
    indices, sentences = zip(*map_index2sent.items())

    if map_action is not None:
        answers = ['Bye bye! Thanks for the chat!',
                   'Hi, I am good. What about you?',
                   'I am a bot designed to solve computer problems.',
                   'I can help you with computer troubleshooting related questions.',
                   'Sorry, I cant help you with that.',
                   'What version of windows do you have?',
                   'Please follow this solution: https://www.dell.com/support/kbdoc/en-us/000132142/windows-10-crashes-to-a-blue-screen']
        sentences = answers
    EMB = embeddings([], embed_par='tf')
    emb = EMB.fit(sentences)
    return dict(zip(indices, emb))


def warmup_run(agent, simulator, params, mapping, use_Q = True, verbose=True):
    Q_table = Q_tab(mapping)

    if use_Q:
        Q_table.load_Q()
        
    num_episodes = params['num_episodes_warm']
    num_iter_warm = 1

    map_state2index = mapping['state2index']
    map_index2action = mapping['index2action']
    map_action2index = mapping['action2index']

    for iter in range(num_iter_warm):
        conv = simulator.sequential(reset=True)
        utterance = conv['utterance']
        episode_max = max(num_episodes, len(map_state2index))
        
        for episode in range(episode_max):
            s_t = utterance

            if verbose:
                print('%%%%%%%%%%%%%%%%%%%%%%%%')
                print('utterance:', utterance)
            
            if use_Q:
                index_state = map_state2index[s_t]
                index_action, confidence = Q_table.predict(index_state)
                intent = 'Unknown'
                action = map_index2action[index_action]
            else:
                action, intent, confidence = agent.rule_policy(utterance)

            reward = confidence
            index_state = map_state2index[s_t]
            index_action = map_action2index[action]

            if not use_Q:
                Q_table.add(confidence, index_state, index_action)

            conv = simulator.sequential()

            if conv is not None:
                utterance = conv['utterance'] #can be different from 1 in slots
                s_t_plus1 = utterance
                episode_over = True

                if verbose:
                    print('episode #', episode)
                    print('s_t:', s_t, map_state2index[s_t])
                    print('action:', action, map_action2index[action])
                    print('intent:', intent)
                    print('s_t_plus1:', s_t_plus1)
                    print('reward:', reward)
                    print()
                fill_buffer_all_actions(agent, s_t, action, reward, s_t_plus1, episode_over, warmup=True)
            else:
                break

    if not use_Q:
        Q_table.save_Q()

    print('WARM-UP TRAINING ....')
    agent.clone_dqn = copy.deepcopy(agent.dqn)
    agent.train(30,50, num_iter = 100)
    return agent, Q_table


def testing(agent, mapping, Q_table, from_Q_table = True, verbose=True, num_test=-1):
    map_index2state = mapping['index2state']
    map_index2action = mapping['index2action']
    map_action2index = mapping['action2index']

    if num_test < 0 :
        indices = range(agent.user_act_cardinality)
    else:
        indices = range(num_test)

    num_success = 0

    for index in indices:
        example = map_index2state[index]
        rep = np.zeros((1,len(map_index2state)))
        rep[0, index] = 1.0
        index_action_dqn = agent.dqn.predict(rep, {'gamma': agent.gamma})

        if from_Q_table:
            index_watson, _ = Q_table.predict(index)
            action_watson = map_index2action[index_watson]
        else:
            action_watson,_,_ = agent.rule_policy(example)
            index_watson = map_action2index[action_watson]

        if index_watson - index_action_dqn == 0:
            num_success += 1

        if verbose:    
            print('------------------------------------------------------')
            print('input:', example, index)
            print('action NLU:', action_watson, index_watson)
            print('action DQN:', map_index2action[index_action_dqn])
            print('------------------------------------------------------')

    print('In warm-up: # success %d, success rate %f' %( num_success, float(num_success)/float(len(indices))))
    return


def fill_buffer_all_actions(agent, s_t, action, reward, s_t_plus1, episode_over, warmup=False, reward_thre = 0.5, flush=False):
    agent.register_experience_replay_tuple(s_t, action, reward,  s_t_plus1, episode_over, flush=flush)
    do_augmentation = False
    
    if warmup:
        do_augmentation = True
    else:
        if reward > reward_thre:
            do_augmentation = True
          
    if do_augmentation:
        for act in agent.map_action2index.keys():
            if act != action:
                zero_reward = 0. 
                agent.register_experience_replay_tuple(s_t, act, zero_reward,  s_t_plus1, episode_over, flush=False)
    return


def save_model(path, agent, epoch):
    """ save only model params need to be added """
    filename = 'agt_%d.p' % (epoch)
    filepath = os.path.join(path, filename)
    checkpoint = {}
    checkpoint['model'] = copy.deepcopy(agent.dqn.model)
    try:
        pickle.dump(checkpoint, open(filepath, "wb"))
        print('saved model in %s' % (filepath, ))
    except Exception as e:    
        print('Error: Writing model fails: %s' % (filepath, ))
        print(e)

        
