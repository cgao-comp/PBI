from utils import *
from BayesianInference import *
from dani import *

if __name__ == '__main__':

    RG = load_pickle('./G.pkl')
    cascades = read_cascades('beta_500.txt')

    N = len(RG.nodes)
    M = len(cascades)

    H = 0.2

    print("---------------start pre-pruning work--------------")
    cut_IG, result, A = DANI(N, cascades, K=(int(N * (N - 1) * 0.5 * H)))
    all_cands = []
    for (u,v) in cut_IG.edges:
        if u < v:
            all_cands.append((u,v))
        else:
            all_cands.append((v,u))

    start_time = time.time()
    print("---------------start inference---------------------")
    IG_s = paral_MCMC(N, cascades, alpha=1, p=0.4, iter=2*N**2, error=-M, sample_size=5, cpu_num=5, all_cands = all_cands)
    print("running time : {}".format(round((time.time() - start_time), 2)))

    avg_Presion, avg_Recall, avg_F_score = 0,0,0
    for IG in IG_s:
        p,r,f = compute_F_score(IG,RG,is_directed=False)
        avg_Presion += p
        avg_Recall += r
        avg_F_score += f


    print("Inference result : Precision : {}, Recall : {}, F-score : {}".format(round(avg_Presion/len(IG_s),2),
                                                                                round(avg_Recall/len(IG_s),2),
                                                                                round(avg_F_score/len(IG_s),2)))