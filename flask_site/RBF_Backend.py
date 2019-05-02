from datetime import datetime
import csv
from numpy import *
from math import sqrt

def replaceLingua(dataset, index):
    data = []
    for line in dataset:
        data.append(line[index])

    return unique(data)

def cardinal_to_degrees(cardinal):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    if cardinal == "NA" or cardinal == 0:
       return 0
    else:
       return (dirs.index(cardinal)*22.5)
       
def load_data():
    #with open('weatherAUS.csv') as csv_file:
    with open('weatherFixed.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        feature_data = []
        location = []
        _location = []
        labels = [] 
        i = 0
        for line in csv_reader:
            if i != 0:
                feature_tmp = []            
                for value in line:
                    if value == 'NA':
                        feature_tmp.append(0)
                    else:
                        feature_tmp.append(value)                
            
                #labels for classes
                labels.append(feature_tmp[len(feature_tmp) - 1])
                indexesToRemove = [0, 5, 6, 17, 18, 21, 22, 23]
                for index in sorted(indexesToRemove, reverse=True):
                    feature_tmp.pop(index)
                    
                feature_data.append(feature_tmp)
            i += 1
            #print(feature_tmp)
        #print(len(feature_data))        

        
        _location = replaceLingua(feature_data, 0)
        feature_data.pop(0)
        for line in feature_data:
            line[0] = _location.tolist().index(line[0])            
            line[4] = float(cardinal_to_degrees(line[4]))
            line[6] = float(cardinal_to_degrees(line[6]))
            line[7] = float(cardinal_to_degrees(line[7]))           
            for indexer in range(0, len(line)): 
                line[indexer] = float(line[indexer])
        #print(labels)
        for indexer in range(0, len(labels)): 
            if labels[indexer] == 'Yes':
                labels[indexer]=1
            else:
                labels[indexer]=0
        n_output = 1
        labels.pop(len(labels)-1)

    return mat(feature_data), mat(labels).transpose(), n_output


def linear(x):
    return x


def hidden_out(feature, center, delta):
    m, n = shape(feature)
    m1, n1 = shape(center)
    hidden_out = mat(zeros((m, m1)))
    for i in range(m):
        for j in range(m1):
            #gaus
            hidden_out[i, j] = exp(-1.0 * (feature[i, :] - center[j, :]) * (feature[i, :] - center[j, :]).T / (
                        2 * delta[0, j] * delta[0, j]))
    return hidden_out


def predict_in(hidden_out, w):
    m = shape(hidden_out)[0]
    predict_in = hidden_out * w
    return predict_in


def predict_out(predict_in):
    result = linear(predict_in)
    return result


def bp_train(feature, label, n_hidden, maxCycle, alpha, n_output):
    m, n = shape(feature)
    center = mat(random.rand(n_hidden, n))
    center = center * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - mat(ones((n_hidden, n))) * (
                4.0 * sqrt(6) / sqrt(n + n_hidden))
    delta = mat(random.rand(1, n_hidden))
    delta = delta * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - mat(ones((1, n_hidden))) * (
                4.0 * sqrt(6) / sqrt(n + n_hidden))
    w = mat(random.rand(n_hidden, n_output))
    w = w * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - mat(ones((n_hidden, n_output))) * (
                4.0 * sqrt(6) / sqrt(n_hidden + n_output))

    iter = 0
    while iter <= maxCycle:
        hidden_output = hidden_out(feature, center, delta)
        output_in = predict_in(hidden_output, w)
        output_out = predict_out(output_in)
        error = mat(label - output_out)
        for j in range(n_hidden):
            sum1 = 0.0
            sum2 = 0.0
            sum3 = 0.0
            for i in range(m):
                sum1 += error[i, :] * exp(
                    -1.0 * (feature[i] - center[j]) * (feature[i] - center[j]).T / (2 * delta[0, j] * delta[0, j])) * (
                                    feature[i] - center[j])
                sum2 += error[i, :] * exp(
                    -1.0 * (feature[i] - center[j]) * (feature[i] - center[j]).T / (2 * delta[0, j] * delta[0, j])) * (
                                    feature[i] - center[j]) * (feature[i] - center[j]).T
                sum3 += error[i, :] * exp(
                    -1.0 * (feature[i] - center[j]) * (feature[i] - center[j]).T / (2 * delta[0, j] * delta[0, j]))
            delta_center = (w[j, :] / (delta[0, j] * delta[0, j])) * sum1
            delta_delta = (w[j, :] / (delta[0, j] * delta[0, j] * delta[0, j])) * sum2
            delta_w = sum3
            center[j, :] = center[j, :] + alpha * delta_center
            delta[0, j] = delta[0, j] + alpha * delta_delta
            w[j, :] = w[j, :] + alpha * delta_w
        if iter % 10 == 0:
            cost = (1.0 / 2) * get_cost(get_predict(feature, center, delta, w) - label)
            print("\t-------- iter: ", iter, " ,cost: ", cost)
        if cost < 3:
            break
        iter += 1
    return center, delta, w


def get_cost(cost):
    m, n = shape(cost)

    cost_sum = 0.0
    for i in range(m):
        for j in range(n):
            cost_sum += cost[i, j] * cost[i, j]
    return cost_sum / 2


def get_predict(feature, center, delta, w):
    return predict_out(predict_in(hidden_out(feature, center, delta), w))


def save_model_result(center, delta, w, result):
    def write_file(file_name, source):
        f = open(file_name, "w")
        m, n = shape(source)
        for i in range(m):
            tmp = []
            for j in range(n):
                tmp.append(str(source[i, j]))
            f.write("\t".join(tmp) + "\n")
        f.close()

    write_file("messidor_center.txt", center)
    write_file("messidor_delta.txt", delta)
    write_file("messidor_weight.txt", w)
    write_file('messidor_train_result.txt', result)


def err_rate(label, pre):
    m = shape(label)[0]
    for j in range(m):
        if pre[j, 0] > 0.5:
            pre[j, 0] = 1.0
        else:
            pre[j, 0] = 0.0

    err = 0.0
    for i in range(m):
        if float(label[i, 0]) != float(pre[i, 0]):
            err += 1
    rate = err / m
    return rate

def load_model(file_center, file_delta, file_w):
    def get_model(file_name):
        f = open(file_name)
        model = []
        for line in f.readlines():
            lines = line.strip().split("\t")
            model_tmp = []
            for x in lines:
                model_tmp.append(float(x.strip()))
            model.append(model_tmp)
        f.close()
        return mat(model)

    center = get_model(file_center)

    delta = get_model(file_delta)

    w = get_model(file_w)

    return center, delta, w

print("--------- 1.load data ------------")
feature, label, n_output = load_data()
print("--------- 2.training ------------")
#center, delta, w = bp_train(feature, label, 20, 5000, 0.008, n_output)
center, delta, w = bp_train(feature, label, 20, 5, 0.008, n_output)
print("--------- 3.get prediction ------------")
result = get_predict(feature, center, delta, w)
print("result：", (1 - err_rate(label, result)))
print("--------- 4.save model and result ------------")
save_model_result(center, delta, w, result)