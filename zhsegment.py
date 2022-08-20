import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10, log2

DIGITS = {"１", "２", "３", "４", "５", "６", "７", "８", "９", "０", "·", "．", "第"}
C_DIGITS = {"一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万", "亿", "第", "点", "·", "．"}
NUMBERS = {"一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万", "亿", "１", "２", "３", "４", "５", "６", "７", "８", "９", "０", "第", "点", "·", "．"}
UNITS = {"十", "百", "千", "万", "亿", "日", "月"}

class Entry:
    def __init__(self, prev, word, spos, logp, bptr):
        self.prev = prev
        self.word = word
        self.spos = spos
        self.logp = logp
        self.bptr = bptr
    def __str__(self):
        return "prev: {pw}, word: {w}, start position: {sp}, log probability: {logp}, back pointer: {bptr}".\
            format(pw = self.prev, w = self.word, sp = self.spos, logp = self.logp, bptr = self.bptr)
    def __lt__(self, other):
        return -self.logp < -other.logp

class Segment:

    def __init__(self, Pw):
        self.Pw = Pw

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."

        chart = {}
        wordset = set()
        heap = []
        for i in range(min(len(text), 13)):
            w = text[:i+1]
            pw = math.log(self.Pw(("<S>", w)))
            e = Entry("<S>", w, 0, pw, None)
            heapq.heappush(heap, (-pw, e))
            wordset.add(('<S>', w))
            # wordset.add(w)
        
        while(len(heap) > 0):
            e = heapq.heappop(heap)[1]
            # wordset.remove(e.word)
            wordset.remove((e.prev, e.word))
            # print(e)
            epos = e.spos + len(e.word)
            if epos in chart:
                if chart[epos].logp < e.logp:
                    chart[epos] = e
                else:
                    if e.word == chart[epos].word:
                        continue
            else:
                chart[epos] = e

            # start = epos
            # end = epos + min(len(text) - epos, 13)
            # print(str(start) + " : " + str(end))
            for i in range(epos, epos + min(len(text) - epos, 13)):
                nw = text[epos:i+1]
                # print("Searching word: " + nw)
                if (e.word, nw) not in wordset:
                    npw = math.log(self.Pw((e.word, nw))) + e.logp
                    ne = Entry(e.word, nw, epos, npw, e.spos)
                    heapq.heappush(heap, (-npw, ne))
                    wordset.add((e.word, nw))
                
        # for k in chart:
        #     print("chart [{k}] = ".format(k = k) + str(chart[k]))
        segmentation = []
        epos = len(text)
        while(epos > 0):
            e = chart[epos]
            segmentation.insert(0, e.word)
            epos = e.spos
        temp = []
        for word in segmentation:
            word = re.split(r"(。|？|！|，|、|；|：|「|」|『|』|‘|’|“|”|（|）|〔|〕|【|】|—|…|–|《|》|〈|〉|的)", word)
            for i in word:
                i.replace(" ", "")
            word = list(filter(None, word))
            temp.extend(word)
        segmentation = [temp[0]]

        for word in temp[1:]:
            if(word[0] == "·" or segmentation[-1][-1] == "·"):
                segmentation[-1] += word
            else:
                if(word in UNITS and all([c in NUMBERS for c in segmentation[-1]])):
                    segmentation[-1] += word
                else:
                    segmentation.append(word)

        return segmentation

    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)

#### Support functions (p. 224)

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

# class Pdist(dict):
#     "A probability distribution estimated from counts in datafile."
#     def __init__(self, data=[], N=None, missingfn=None):
#         for key,count in data:
#             self[key] = self.get(key, 0) + int(count)
#         self.N = float(N or sum(self.values()))
#         self.missingfn = missingfn or avoid_long_words
#     def __call__(self, key): 
#         if key in self: return self[key]/self.N  
#         else: return self.missingfn(key, self.N)

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, unigram=[], bigram=[], N=None, missingfn=None):
        self.unigramN = dict()
        count_s = 0
        for key,count in unigram:
            self.unigramN[key] = self.unigramN.get(key, 0) + int(count)
        for key,count in bigram:
            (prev, cur) = key.split(' ')
            # print((prev, cur))
            self[(prev, cur)] = self.get((prev, cur), 0) + int(count)
            if prev == '<S>':
                count_s += int(count)
        
        self.unigramN['<S>'] = count_s
        self.N = sum(self.unigramN.values())
        self.missingfn = missingfn or avoid_long_words
    def __call__(self, key):
        if len(key[1]) > 1 and (all([c in DIGITS for c in key[1]]) or all([c in C_DIGITS for c in key[1]])):
            return 1/self.N
        if key in self: 
            return 0.99 * self[key]/float(self.unigramN[key[0]]) + 0.01*self.unigramN[key[1]]/self.N
        else:
            if key[1] in self.unigramN:
                 return self.unigramN[key[1]]/self.N
            else:
                return self.missingfn(key[1], self.N)

def avoid_long_words(word, N):
    "Estimate the probability of an unknown word."
    # print(word)
    # if len(word) > 4:
    #     return 10. / (N * 10000 ** len(word))
    return 10. / (N * 1000 ** len(word))

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    Pw = Pdist(unigram=datafile(opts.counts1w), bigram=datafile(opts.counts2w))
    segmenter = Segment(Pw)
    with open(opts.input) as f:
        for line in f:
            print(" ".join(segmenter.segment(line.strip())))
