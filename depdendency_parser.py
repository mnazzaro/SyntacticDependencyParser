from cmath import e
import numpy as np
import math
import functools
from typing import Tuple, List, Dict
import pandas as pd
from tqdm import tqdm

def softmax (v: np.array) -> np.array:
    base = functools.reduce(lambda a, b: a + b, map(lambda x: math.e ** x, v))
    return np.array(list(map(lambda x: (e ** x) / base, v)))

def signum (x: int) -> int:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0

def perceptron (x: np.array, W: np.array, b: int) -> int:
    return signum(np.dot(x, W) + b)

# Take labels and max iterations, return W and b
def perceptron_train (D: List[Tuple[np.array, int]], iterations: int) -> Tuple[np.array, int]:
    W = np.zeros(len(D[0][0]))
    b = 0
    for i in range(iterations):
        for x, y in D:
            a = perceptron(x, W, b)
            if y * a <= 0:
                W += x * y
                b += y
    return (W, b)

D = []
D.append((np.array([1, 0, 0, 0]), -1))
D.append((np.array([0, 0, 1, 0]), -1))
D.append((np.array([0, 0, 0, 1]), 1))
D.append((np.array([0, 1, 0, 0]), 1))

(W, b) = perceptron_train(D, 5)
# print (W)
# print (b)


def test (D: List[Tuple[np.array, int]], W: np.array, b: int) -> bool:
    for x, y in D:
        if perceptron (x, W, b) != y:
            return False
    return True

# print (test(D, W, b))


#######################################################

################ Syntactic Structure ##################

#######################################################

class DependencyTree:

    def __init__ (self):
        self.length = 0
        self.words = {0 : ("ROOT", "ROOT")}
        self.dependencies = {0 : None}

    def add_dependency (self, head: int, dependent: int, dtype: np.array):
        self.dependencies[dependent] = (head, dtype)

    def find_word (self, word):
        ret = 0
        for i in self.words:
            if word == i:
                return ret
            ret += 1
        return -1

    def add_word (self, word: str, pos: np.array):
        self.length += 1
        self.words[self.length] = (word, pos)

    def get_words (self):
        ret = []
        for i in range(1, self.length+1):
            ret.append((i, self.words[i]))
        return ret

    #TODO: Add validity checker for dependencies


###### Generate training examples ######

# A parse is a list of actions

def shift (words, stack):
    stack.append(words.pop(0))

def left_arc (stack):
    stack.pop(-2)

def right_arc (stack):
    stack.pop()

def right_arc_condition(word, tree, words):
    twords = tree.get_words()
    for i in range(word[0], len(twords)):
        if tree.dependencies[twords[i][0]][0] == word[0] and (twords[i] in words):
            return False
    return True


def tree_2_parse (tree: DependencyTree):
    result = []
    stack = [(0, ('ROOT', 'ROOT'))]
    words = tree.get_words()
    while len(words) > 0 or len(stack) != 1:
        if len(stack) < 2:
            shift(words, stack)
            result.append("Shift")
        elif stack[-2][1][0] != 'ROOT' and tree.dependencies[stack[-2][0]][0] == stack[-1][0]:
            left_arc(stack)
            result.append("LeftArc")
        elif stack[-1][1][0] != 'ROOT' and tree.dependencies[stack[-1][0]][0] == stack[-2][0] and right_arc_condition(stack[-1], tree, words):
            right_arc(stack)
            result.append("RightArc")
        else:
            shift(words, stack)
            result.append("Shift")
    return result

"""dependencies = {
    1: (0, 0),
    2: (1, 0),
    3: (5, 0),
    4: (5, 0),
    5: (1, 0)
}

example = DependencyTree()
example.dependencies = dependencies
example.add_word("Book", 0)
example.add_word("me", 0)
example.add_word("the", 0)
example.add_word("morning", 0)
example.add_word("flight", 0)
parse = tree_2_parse(example)
for i in parse:
    print (i)"""

###### Interface with Universal Dependency dataset ######

#sample is a conllu sample from the UD dataset
def ud_2_tree (sample: str, pos_map: Dict[str, np.array], dtype_map: Dict[str, np.array]) -> DependencyTree:
    lines = sample.split("\n")
    tree = DependencyTree()
    for line in lines:
        if len(line) > 0 and (not line[0] == '#'):
            parts = line.split('\t')
            if not parts[0].isnumeric():
                return None
            tree.add_word(parts[1], parts[4])
            tree.add_dependency(int(parts[6]), tree.length, dtype_map[parts[7]])
    #TODO: Check to make sure dependency tree is valid
    return tree

dataset_path = "C://Users//markn//Downloads//Universal Dependencies 2.10//ud-treebanks-v2.10//ud-treebanks-v2.10//UD_English-EWT//en_ewt-ud-train.conllu"
train = []
fails = []
words = []
pos = []
dtypes = []
with open(dataset_path, encoding='utf-8') as f:
    sample = ""
    line = " "
    while line:
        line = f.readline()
        if len(line) > 1 and line[0] != '#':
            l = line.split('\t')
            pos.append(l[4])
            dtypes.append(l[7])
            words.append(l[2])
    f.close()

pos = set(pos)
pos_map = {}
num_pos = len(pos)

pos_map["ROOT"] = np.zeros(num_pos + 1)
pos_map["ROOT"][0] = 1
for i, part in enumerate(pos):
    arr = np.zeros(num_pos + 1)
    arr[i+1] = 1
    pos_map[part] = arr

dtypes = set(dtypes)
dtype_map = {}
num_dtype = len(dtypes)

for i, dtype in enumerate(dtypes):
    arr = np.zeros(num_dtype)
    arr[i] = 1
    dtype_map[dtype] = arr

words = set(words)
word_map = {}
num_words = len(words)
# word_map["ROOT"] = np.zeros(num_words + 1)
# word_map["ROOT"][0] = 1
# for i, word in enumerate(words):
#     arr = np.zeros(num_words+1)
#     arr[i+1] = 1
#     word_map[word] = arr

with open(dataset_path, encoding='utf-8') as f:
    cur_sample = ""
    line = " "
    while line:
        line = f.readline()
        if line == "\n":
            t = ud_2_tree(sample, pos_map, dtype_map)
            if t:
                try:
                    train.append((t, tree_2_parse(t)))
                except:
                    fails.append((t, None))
            sample = ""
        else:
            sample += line
t = ud_2_tree(sample, pos_map, dtype_map)
if t:
    train.append((t, tree_2_parse(t)))

for (t, _) in train:
    for (_, (word, _)) in t.get_words():
        words.add(word)

num_words = len(words)
word_map["ROOT"] = np.zeros(num_words + 1)
word_map["ROOT"][0] = 1
for i, word in enumerate(words):
    arr = np.zeros(num_words+1)
    arr[i+1] = 1
    word_map[word] = arr


# print (len(train))
# print (fails[0][0].get_words())

# score function (useless and incorrect lol)
def score (tree: DependencyTree, W: np.array) -> int:
    res = 0
    for i in extract_features(tree):
        res += np.dot(W, i)
    return res

###### extracting feature functions ######

def extract_features (tree: DependencyTree, stack: List, buffer: List) -> np.array:
    stack0 = stack[-1]
    res = word_map[stack0[1][0]]
    res = np.append(res, pos_map[stack0[1][1]])
    if len(stack) > 1:
        res = np.append(res, 1)
        stack1 = stack[-2]
        res = np.append(res, word_map[stack1[1][0]])
        res = np.append(res, pos_map[stack1[1][1]])
    else:
        res = np.append(res, 0)
        res = np.append(res, np.zeros(num_words + 1 + num_pos + 1))


    if stack0[0] != 0:
        res = np.append(res, 1)
        res = np.append(res, pos_map[tree.words[stack0[0] - 1][1]])
    else:
        res = np.append(res, 0)
        res = np.append(res, np.zeros(num_pos+1))
    if stack0[0] != tree.length:
        res = np.append(res, 1)
        res = np.append(res, pos_map[tree.words[stack0[0] + 1][1]])
    else:
        res = np.append(res, 0)
        res = np.append(res, np.zeros(num_pos+1))
    if len(stack) > 1:
        res = np.append(res, 1)
        if stack1[0] != 0:
            res = np.append(res, 1)
            res = np.append(res, pos_map[tree.words[stack1[0] - 1][1]])
        else:
            res = np.append(res, 0)
            res = np.append(res, np.zeros(num_pos+1))
        if stack1[0] != tree.length:
            res = np.append(res, 1)
            res = np.append(res, pos_map[tree.words[stack1[0] + 1][1]])
        else:
            res = np.append(res, 0)
            res = np.append(res, np.zeros(num_pos+1))
    else:
        res = np.append(res, 0)
        res = np.append(res, np.zeros(2 * (num_pos + 1) + 2))


    if len(buffer) > 0:
        res = np.append(res, 1)
        res = np.append(res, word_map[buffer[0][1][0]])
    else:
        res = np.append(res, 0)
        res = np.append(res, np.zeros(num_words + 1))

    if len(buffer) > 1:
        res = np.append(res, 1)
        res = np.append(res, word_map[buffer[1][1][0]])
    else:
        res = np.append(res, 0)
        res = np.append(res, np.zeros(num_words + 1))

    return res

###### Logistic Regression ######

import matplotlib.pyplot as plt
def log_reg_train (trees: List[DependencyTree], iterations: int, alpha: float) -> Tuple[np.array, int]:
    feature_len = len(extract_features(trees[0], [(0, ('ROOT', 'ROOT'))], trees[0].get_words()))
    W = np.zeros((3, feature_len + 1))
    #V = np.zeros((3, feature_len))
    y = np.zeros(4)
    losses = []
    correct = 0
    for q in range(iterations):
        loss = 0
        accuracy = 0
        count = 0
        accuracy_t = 0
        count_t = 0
        for i in tqdm(range(len(trees))):
            tree_check = True
            tree = trees[i]
            stack = [(0, ('ROOT', 'ROOT'))]
            words = tree.get_words()
            while len(words) > 0 or len(stack) != 1:
                features = np.append(extract_features(tree, stack, words), 1)
                pred = softmax(np.matmul(W, features))
                y_hat = np.argmax(pred)
                if len(stack) < 2:
                    shift(words, stack)
                    y = np.array([1, 0, 0])
                    correct = 0
                elif stack[-2][1][0] != 'ROOT' and tree.dependencies[stack[-2][0]][0] == stack[-1][0]:
                    left_arc(stack)
                    y = np.array([0, 1, 0])
                    correct = 1
                elif stack[-1][1][0] != 'ROOT' and tree.dependencies[stack[-1][0]][0] == stack[-2][0] and right_arc_condition(stack[-1], tree, words):
                    right_arc(stack)
                    y = np.array([0, 0, 1])
                    correct = 2
                else:
                    shift(words, stack)
                    y = np.array([1, 0, 0])
                    correct = 0
                if y_hat == correct:
                    accuracy += 1
                else:
                    tree_check = False
                loss += -math.log(np.dot(pred, y))
                t = pred - y
                g = np.outer(features, pred-y)
                W -= g.transpose() * alpha
                count += 1 
            count_t += 1
            if tree_check:
                accuracy_t += 1
        losses.append(loss)
        print (f"Iteration {q+1} => TOTAL LOSS: {loss}, INDIVIDUAL ACCURACY: {accuracy / count}, TREE ACCURACY: {accuracy_t / count_t}")
        # plt.plot(list(range(1, q+2)), losses)
        # plt.xlabel("Iteration")
        # plt.ylabel("Total Loss")
        # plt.show()
    return W
    

W = log_reg_train([x[0] for x in train], 7, 1.2)
np.savetxt("weights.gz", W, delimiter=",")
#W = np.loadtxt("weights.gz", delimiter=",")

example = DependencyTree()
example.add_word("Book", "VB")
example.add_word("the", "DT")
example.add_word("flight", "NN")
example.add_word("through", "IN")
example.add_word("Houston", "NNP")

def parse_to_tree (tree: DependencyTree, parse: List[str]):
    words = tree.get_words()
    stack = [(0, ('ROOT', 'ROOT'))]
    for action in parse:
        if action == "SHIFT":
            shift(words, stack)
        elif action == "LEFT ARC":
            tree.add_dependency(stack[-1][0], stack[-2][0], None)
            left_arc(stack)
        else:
            tree.add_dependency(stack[-2][0], stack[-1][0], None)
            right_arc(stack)

def create_dependencies(tree: DependencyTree):
    stack = [(0, ('ROOT', 'ROOT'))]
    words = tree.get_words()
    result = []
    while len(words) > 0 or len(stack) != 1:
        features = np.append(extract_features(tree, stack, words), 1)
        y_hat = softmax(np.matmul(W, features))
        pred = np.argmax(y_hat)
        if pred == 0:
            shift(words, stack)
            result.append("SHIFT")
        elif pred == 1:
            left_arc(stack)
            result.append("LEFT ARC")
        elif pred == 2:
            right_arc(stack)
            result.append("RIGHT ARC")
    parse_to_tree(tree, result)
    output = "HEAD\tDEPENDENT\n"
    for i in tree.dependencies.keys():
        if i != 0:
            output+=tree.words[tree.dependencies[i][0]][0]
            output+="\t"
            output+=tree.words[i][0]
            output+="\n"
    return output

print (create_dependencies(example))
