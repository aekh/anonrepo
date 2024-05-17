from alphamart import *
from awaire_utils import *
from timeit import default_timer as timer
import os  # Do not remove
import sys
import gc


class AlphaMart(TestMartingale):
    def __init__(self, N, d, eta0, cvrs=None):
        super().__init__()
        self.N = N
        if cvrs is None:
            self.eta = eta0
        self.d = d
        self.j = 0
        self.S = 1
        self.mu = 0.5
        # params (S0, j0, eta0)

    def init_params(self) -> tuple:
        # TODO include CVRs here
        return 0, 1, self.eta

    def process(self, data: np.ndarray, params: tuple) -> (np.ndarray, tuple):
        increments, params = self.test_list(data, *params)
        return increments, params

    def test_list(self, x, S0, j0, eta0):
        eta0 = 0.52 if eta0 <= 0.5 else eta0
        eta_i, mu_i = shrink_trunc(x, N=self.N, nu=eta0, d=self.d, S0=S0, j0=j0, c=(eta0-0.5)/2)
        res, _ = alpha_mart(np.array(x), mu_i, eta_i)
        # print("   ", np.ma.log(res).filled(-np.inf), (S0, j0, x), (S0 + sum(x), j0 + len(x), eta0))
        return np.ma.log(res).filled(-np.inf), (S0 + sum(x), j0 + len(x), eta0)


def read_election_files(source, d, perm, candincr=0):
    US_TAGS = "CityCouncil", "Mayor", "CountyAssessor", "CountyExecutive", "CountyAuditor"
    memb = lambda x, y: x in y
    if any(memb(i, d) for i in US_TAGS):
        prefix = "USIRV/"
        suffix = ""
    else:
        prefix = "NSW2015/Data_NA_"
        suffix = ".txt_ballots"
    ballotnumberdiff = -1
    if "pathological" in d:
        prefix = "pathological/"
        suffix = ""
        ballotnumberdiff = 0  # hotfix

    ballotfile = source + prefix + d + suffix + ".txt"
    marginfile = source + "margins/" + prefix + d + suffix + ".csv"
    orderfile = source + "orderings/" + prefix + d + suffix + ".csv"
    ballotdata = []

    with open(ballotfile, "r") as file:
        line = file.readline()
        candmap = dict()
        candlist = [i.strip() for i in line.split(",")]
        for i, cand in enumerate(candlist):
            candmap[cand] = i
            
        ncand = len(candlist) + candincr
        permuter = perm_decode(perm, ncand)
        population = Ballots(ncand)

        file.readline()
        file.readline()
        for line in file:
            f = line.split(" : ")
            # if len(f[0]) == 2: continue  # Handle empty/invalid ballots
            strballot = f[0].split("(")[1].split(")")[0].split(",")
            if len(strballot) == 2 and strballot[1] == '':
                strballot = [strballot[0]]  # compatibility issue fix

            if len(strballot) == 1 and strballot[0] == '':
                ballot_true = []
                ballot = []
            else:
                ballot_true = [candmap[i.strip()] for i in strballot]
                ballot = [permuter[candmap[i.strip()]] for i in strballot]  # Error model
            # code = population.to_code(ballot)
            # code_true = population.to_code(ballot_true)
            # ballotdata += [code_true] * int(f[1])
            # for i in range(int(f[1])):
            #     population.observe(code)
            ballotdata += [tuple(ballot_true)] * int(f[1])
            for i in range(int(f[1])):
                population.observe(tuple(ballot))
    nballots = len(ballotdata)
    margindata = [None] * population.ncand
    nomargin = False
    try:
        with open(marginfile, "r") as file:
            csv_reader = csv.reader(file, delimiter=',')
            for row in csv_reader:
                if csv_reader.line_num == 1: continue
                margindata[candmap[row[1].strip()]] = int(row[2]) / len(ballotdata)
    except FileNotFoundError:
        nomargin = True
    orderdata = []
    with open(orderfile, "r") as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if csv_reader.line_num == 1: continue
            orderdata.append([int(i) for i in row])
    margindata = np.array(margindata)
    margindata[margindata==None] = -1 # TODO hotfix

    fakesequence, last_round_margin = population.simulate()

    if nomargin:
        reported = fakesequence[-1]
        margindata[reported] = last_round_margin - 0.5

    truewin = np.argmax(margindata)

    return ncand, nballots, ballotdata, margindata, orderdata, fakesequence, ballotnumberdiff, truewin, last_round_margin, population


def func():
    rng = np.random.default_rng()
    datafiles = [
        # NSW 2015
        "Albury","Auburn","Ballina","Balmain","Bankstown","Barwon","Bathurst","Baulkham_Hills","Bega","Blacktown",
        "Blue_Mountains","Cabramatta","Camden","Campbelltown","Canterbury","Castle_Hill","Cessnock","Charlestown",
        "Clarence","Coffs_Harbour","Coogee","Cootamundra","Cronulla","Davidson","Drummoyne","Dubbo","East_Hills",
        "Epping","Fairfield","Gosford","Goulburn","Granville","Hawkesbury","Heathcote","Heffron","Holsworthy","Hornsby",
        "Keira","Kiama","Kogarah","Ku-ring-gai","Lake_Macquarie","Lakemba","Lane_Cove","Lismore","Liverpool",
        "Londonderry","Macquarie_Fields","Maitland","Manly","Maroubra","Miranda","Monaro","Mount_Druitt","Mulgoa",
        "Murray","Myall_Lakes","Newcastle","Newtown","Northern_Tablelands","North_Shore","Oatley","Orange","Oxley",
        "Parramatta","Penrith","Pittwater","Port_Macquarie","Port_Stephens","Prospect","Riverstone","Rockdale","Ryde",
        "Seven_Hills","Shellharbour","South_Coast","Strathfield","Summer_Hill","Swansea","Sydney","Tamworth","Terrigal",
        "The_Entrance","Tweed","Upper_Hunter","Vaucluse","Wagga_Wagga","Wakehurst","Wallsend","Willoughby",
        "Wollondilly","Wollongong","Wyong",
        # USIRV
        "Aspen_2009_CityCouncil","Berkeley_2010_D1CityCouncil","Berkeley_2010_D7CityCouncil",
        "Oakland_2010_D4CityCouncil","Oakland_2010_Mayor","Pierce_2008_CountyAssessor","Pierce_2008_CountyExecutive",
        "Aspen_2009_Mayor","Berkeley_2010_D4CityCouncil","Berkeley_2010_D8CityCouncil","Oakland_2010_D6CityCouncil",
        "Pierce_2008_CityCouncil","Pierce_2008_CountyAuditor","SanFran_2007_Mayor",
        # USIRV MINNEAPOLIS
        "Minneapolis_2013_Mayor", "Minneapolis_2017_Mayor", "Minneapolis_2021_Mayor"
    ]

    deta = [(200, 0.51, 0.5), (200, None, 0.5), (200, "AM", 0.5)]

    source = "../datafiles/"

    print("dataset, setting, reported, true, margin, candidates, weightmode, d, eta, expcond, alpha, certified, " +
          "trialnum, samplesize, samplelimit, dur, stepsize, " +
          "max_nodes, tot_nodes, avg_nodes, tot_prune, avg_prune_depth, max_depth, max_reqs, tot_reqs, avg_reqs, " +
          "max_req_parked, avg_req_parked, tot_req_prune")

    # increments = np.arange(0, 51)
    increments = [0]

    counter = 0  # 1-4992
    perm = 0  # no errors in CVRs
    for (d, candincr) in ((datafile, candincr) for datafile in datafiles for candincr in increments): # NEXT double loop generator here
        counter = run_contest(candincr, counter, d, deta, perm, source)

        # c = 0
        # w = 0
        # ru = 0
        # if d == "Strathfield":
        #     w = 2
        #     ru = 4
        #     c = 5
        # elif d == "Ballina":
        #     w = 4
        #     ru = 2
        #     c = 7
        # for perm in range(1, np.math.factorial(c)):
        #     seq = perm_decode(perm, c)
        #     last_round_flip = list(range(c))
        #     last_round_flip[w] = ru
        #     last_round_flip[ru] = w
        #     if seq[w] == w or seq == last_round_flip:
        #         run_contest(candincr, counter, d, deta, perm, source)
        #         counter += 2


def run_contest(candincr, counter, d, deta, perm, source):
    settings = [  # "Every40",
        # "Every25",
        # "Every10",
        "IfNegative",
        Largest()
    ]
    for setting in settings:
        for (alpha_d, eta, thres) in deta:
            runsets = [range(100), range(100,200)]

            for runset in runsets:
                counter += 1
                # if counter != int(os.environ['SLURM_ARRAY_TASK_ID']): continue # FIXME

                ncand, nballots, ballotdata, margindata, orderdata, csv_elim_seq, ballotnumberdiff, truewin, \
                    last_round_mean, population = read_election_files(source, d, perm, candincr=candincr)

                if eta is None:
                    eta = last_round_mean

                w = csv_elim_seq[-1]

                margin = margindata[w]

                contest = Contest("Contest", ncand, w, nballots)
                audit = Audit(contest, AlphaMart(nballots, alpha_d, eta), **{"risklimit": 0.05})

                if setting == "Every25":
                    s = Largest()
                    settings = dict({"req_parking": 1, "req_pruning": 2, "node_full_start": False,
                                     "node_expand_every": 25, "node_expand_threshold": -np.inf})
                elif setting == "Every10":
                    s = Largest()
                    settings = dict({"req_parking": 1, "req_pruning": 2, "node_full_start": False,
                                     "node_expand_every": 10, "node_expand_threshold": -np.inf})
                elif setting == "Every40":
                    s = Largest()
                    settings = dict({"req_parking": 1, "req_pruning": 2, "node_full_start": False,
                                     "node_expand_every": 40, "node_expand_threshold": -np.inf})
                elif setting == "IfNegative":
                    s = Largest()
                    settings = dict({"req_parking": 1, "req_pruning": 2, "node_full_start": False,
                                     "node_expand_every": np.inf, "node_expand_threshold": 0})
                else:
                    s = setting
                    settings = dict({"req_parking": 1, "req_pruning": 0, "node_full_start": True})
                    settings["req_no_dnds"] = True

                settings["verbose"] = 0
                settings["node_expand_condition"] = thres
                if eta == "AM":
                    settings["cvrs"] = population

                audit.reset()
                for r in runset:
                    drawnumber = 0
                    start = timer()
                    frontier = Frontier(audit, s, **settings)

                    frontier.create_frontier()

                    #  0: continue,  1: certify,  -1: escalate
                    action = 0
                    while audit.samplesize() < nballots:
                        sample = ballotdata[orderdata[r][drawnumber] + ballotnumberdiff]
                        drawnumber += 1
                        audit.observe(sample)
                        # audit.observe(audit.ballots.to_code([0]))
                        # print("NEW OBSERVATION  ~~~   ", "# ballots:", audit.samplesize(), " pruned requirements:", len(frontier.reqs.pruned), " parked requirements:", len(frontier.reqs.parked))
                        frontier.process_ballots()
                        action = frontier.process_nodes()
                        if action in [-1, 1]:
                            break
                    end = timer()
                    dur = end - start
                    # if done:
                    #     # print("CERTIFIED!")
                    # else:
                    #     # print("FULL RECOUNT REACHED! Did not certify")
                    # print("Ballots,  Operations,  Avg Operations/Ballot")
                    if action == -1:
                        samplesize = nballots
                        certify = False
                    else:
                        samplesize = audit.samplesize()
                        certify = (action == 1)

                    print(d, setting, w, truewin, margin, ncand, str(frontier.weigher), alpha_d, eta, thres,
                          audit.risklimit, certify, r, samplesize, nballots, dur, 1,
                          frontier.stat_max_nodes, frontier.stat_tot_nodes,
                          frontier.stat_sum_nodes / audit.samplesize(),
                          frontier.stat_tot_prune, frontier.stat_sum_prune_depth / frontier.stat_tot_prune,
                          frontier.stat_max_depth, frontier.reqs.stat_max_reqs, frontier.reqs.stat_tot_reqs,
                          frontier.reqs.stat_sum_reqs / audit.samplesize(), frontier.reqs.stat_max_req_parked,
                          frontier.reqs.stat_sum_req_parked / audit.samplesize(),
                          frontier.reqs.stat_tot_req_prune, sep=", ")

                    del frontier
                    audit.reset()
                    gc.collect()
                # sys.exit()
    return counter


if __name__ == '__main__':
    func()
