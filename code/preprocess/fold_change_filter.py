import numpy as np
import pandas as pd
from scipy import stats

data = pd.read_csv("Result.csv")

def split_data(data, normalize=True):
    # pick train/validation split index
    labels = data.values[:, -1]
    positive_index = np.where(labels==1)[0]
    negative_index = np.where(labels==0)[0]
    distributed_ratio = len(positive_index)/len(negative_index)

    validation_ratio = 0.3
    positive_train_index = positive_index[:int((1-validation_ratio)*len(positive_index))]
    negative_train_index = negative_index[:int((1-validation_ratio)*len(negative_index))]
    positive_test_index = positive_index[int((1-validation_ratio)*len(positive_index)):]
    negative_test_index = negative_index[int((1-validation_ratio)*len(negative_index)):]

    train_index = np.concatenate([positive_train_index, negative_train_index], axis=0)

    test_index = np.concatenate([positive_test_index, negative_test_index], axis=0) 

    np.random.shuffle(train_index)
    np.random.shuffle(test_index) 

#    print("train_index: ", train_index)
#    print("test_index: ", test_index)
    epsilon = 10**-10
    mean = np.mean(data.values[:, 2:-1], axis=0)
    std = np.std(data.values[:, 2:-1].astype(np.float32), axis=0)+epsilon
    x_train = (data.values[train_index, 2:-1] - mean)/std
    y_train = data.values[train_index, -1]
    x_test = (data.values[test_index, 2:-1] - mean)/std
    y_test = data.values[test_index, -1] 

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test 


x_train, y_train, x_test, y_test = split_data(data)
np.save("normalized_ori_x_train.npy", x_train)
np.save("normalizedori_y_train.npy", y_train)
np.save("normalizedori_x_test.npy", x_test)
np.save("normalizedori_y_test.npy", y_test)

'''
# result from paper
control_target = ["hsa-miR-23b-3p", "hsa-miR-29a-3p", "hsa-miR-32-5p", "hsa-miR-92a-3p", 
                "hsa-miR-150-5p", "hsa-miR-200a-3p", "hsa-miR-200c-3p", "hsa-miR-203a", 
                "hsa-miR-320c", "hsa-miR-320d", "hsa-miR-335-5p", "hsa-miR-450b-5p", 
                "hsa-miR-1246", "hsa-miR-1307-5p", "label"]

control_data = data[control_target]

x_train, y_train, x_test, y_test = split_data(control_data)
np.save("control_x_train.npy", x_train)
np.save("control_y_train.npy", y_train)
np.save("control_x_test.npy", x_test)
np.save("control_y_test.npy", y_test)
'''

# folder change fliter
arr = data.values[:, 2:]

def folder_change_filter(arr, top_change=1.2, low_change=0.8, min_fraction=0, max_fraction=1):

    epsilon = 10**-10
    normal = arr[np.where(arr[:-1]==0)[0], :-1]
    abnormal = arr[np.where(arr[:-1]==1)[0], :-1]

    normal_mean = np.mean(normal, axis=0)+epsilon
    abnormal_mean = np.mean(abnormal, axis=0)
    fold_change = abnormal_mean.astype(np.float)/normal_mean.astype(np.float)

    '''
    # t-test:
    normal_std = np.std(normal_mean).astype(np.float32)
    abnormal_std = np.std(abnormal).astype(np.float32)
    nobs1 = np.float32(len(normal))
    nobs2 = np.float32(len(abnormal)) 
    modified_std1 = np.sqrt(np.float32(nobs1)/np.float32(nobs1-1)) * normal_std
    modified_std2 = np.sqrt(np.float32(nobs2)/np.float32(nobs2-1)) * abnormal_std
    (statistic, pvalue) = stats.ttest_ind_from_stats(mean1=normal_mean, std1=modified_std1, nobs1=len(normal), mean2=abnormal_mean, std2=modified_std2, nobs2=len(abnormal))
    print("p_value: ", pvalue.shape)
    '''
    # simple_fold_change = np.simple(fold_change.astype(np.float64))
    # print(simple_fold_change.shape, simple_fold_change)
    pass_top_index = np.where(fold_change > top_change)[0].tolist()
    pass_low_index = np.where(fold_change < low_change)[0].tolist()
    pass_index = [i for i in list(set(pass_top_index+pass_low_index)) if (abnormal_mean[i] != 0)] #and (pvalue[i] < 0.05)]
    #print("p values: ", [pvalue[i] for i in pass_index])
    print(pass_index)
    return arr[:, pass_index], pass_index

filtered_arr, pass_index = folder_change_filter(arr)
print(filtered_arr.shape)
pass_index = pass_index
print("pass_index: ", pass_index)
names = []
for i in pass_index:
    names.append(data.columns[i+2])
    print(data.columns[i+2], end=", ")

names = [data.columns[0], data.columns[1]] + names + [data.columns[-1]]

filtered_data = data[names]
print("columns: ", filtered_data.columns)
x_train, y_train, x_test, y_test = split_data(filtered_data)
np.save("normalized_simple_fold_change_filter_x_train.npy", x_train)
np.save("normalized_simple_fold_change_filter_y_train.npy", y_train)
np.save("normalized_simple_fold_change_filter_x_test.npy", x_test)
np.save("normalized_simple_fold_change_filter_y_test.npy", y_test)






