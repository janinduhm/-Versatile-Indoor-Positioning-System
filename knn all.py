
# importing Libraries
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from numpy import sqrt, square, sum, min, std, mean, delete, round
from matplotlib.pyplot import close, ioff, ion
from pandas import DataFrame, concat, read_csv
from os.path import exists, join
from os import mkdir

# Hyper-parameters
# signal null value
NO_SIGNAL_VALUE = -98
# drop data missing columns
DROP_COLUMNS =["SPACEID" ,"RELATIVEPOSITION", "USERID"]
# screen check for plots
DISPLAY_PLOTS = False
SRC_NULL = 100 # Original Null Value

###########################
#######  Functions  #######
###########################


def load_data(train_fname, val_fname, N, drop_columns=None, dst_null=NO_SIGNAL_VALUE, 
              drop_val=False):

    tic = time() # timer start

    if drop_val:
        data = read_csv(train_fname)
    else:
        training_data = read_csv(train_fname)    
        validation_data = read_csv(val_fname)
        data = concat((training_data, validation_data), ignore_index=True)

    #skip unnecessory columns
    if drop_columns: 
        data.drop(columns=drop_columns, inplace=True)
        
    data = data[data.PHONEID != 17] # Phone 17s data is clearly corrupted. 
    # Split data from labels
    X = data.iloc[:, :N]
    Y = data.iloc[:, N:]
    
    # Change null value to 100
    X.replace(100, dst_null, inplace=True)
    X[X < dst_null] = dst_null
    
    # Remove samples that have less than MIN_WAPS active WAPs    
    # Normalize data between 0 and 1 where 1 is strong signal and 0 is null
    X /= min(X)
    X = 1 - X
    
    toc = time() # timer end
    print("Data Load Timer: %.2f seconds" % (toc-tic)) # function time
    
    return X, Y

def filter_out_min_WAPS(data, labels, num_samples=9):
 
    drop_rows = list()
    for i, x in enumerate(data):
        count = sum(x != NO_SIGNAL_VALUE)
        if count < num_samples:
            drop_rows.append(i)
            
    data_new = delete(data, drop_rows, axis=0)
    lbl_new = delete(labels, drop_rows, axis=0)
        
    return data_new, lbl_new



######################
#######  MAIN  #######
######################
    
if __name__ == "__main__":

    tic = time() # Start program timer
    
    close("all") # Close all previously opened plots
    
    #interactive mode for figues
    ion() if DISPLAY_PLOTS else ioff()
    
    # Load and preprocess data with all methods that are independent of subset.
    data = load_data("trainingData.csv", "validationData.csv", 520, DROP_COLUMNS,
                     dst_null=NO_SIGNAL_VALUE, drop_val=True)    
    X, Y = data                
    
    x_train_o, x_test_o, y_train, y_test = train_test_split(X.values, Y.values, 
                                             test_size=0.2, random_state=0)
    
    # filter out samples withoput active WAP values
    x_train_o, y_train = filter_out_min_WAPS(x_train_o, y_train, 9)
    x_test_o, y_test = filter_out_min_WAPS(x_test_o, y_test, 9)

    y_train = DataFrame(y_train, columns=Y.columns)
    y_test = DataFrame(y_test, columns=Y.columns)
    
    
    # K-Nearest Neighbors with Variance Thresholding
    clf = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree',
                                leaf_size=50, p=1)
    regr = KNeighborsRegressor(n_neighbors=1, algorithm='kd_tree',
                                leaf_size=50, p=1)

    variance_thresh = VarianceThreshold(0.00001)
    x_train = variance_thresh.fit_transform(x_train_o)
    x_test = variance_thresh.transform(x_test_o)

    data_in =  (x_train, x_test, y_train, y_test)


    x_train, x_test, y_train, y_test = data_in 


    fit = clf.fit(x_train, y_train[['FLOOR', 'BUILDINGID']])
    prediction = fit.predict(x_test)
    clf_prediction = DataFrame(prediction, columns=['FLOOR', 'BUILDINGID'])
              

    fit = regr.fit(x_train, y_train[['LONGITUDE', 'LATITUDE']])
    prediction = fit.predict(x_test)
    regr_prediction = DataFrame(prediction, columns=['LONGITUDE', 'LATITUDE'])
    
    prediction = concat((clf_prediction, regr_prediction), axis=1)
    

    building_penalty=50
    floor_penalty=4

    build_missclass = sum(prediction["BUILDINGID"].values != y_test["BUILDINGID"].values)
    
    floor_missclass = sum(prediction["FLOOR"].values != y_test["FLOOR"].values)
    
    # localization error using euclidean method
    x, y  = prediction['LONGITUDE'].values, prediction['LATITUDE'].values
    x0, y0 = y_test['LONGITUDE'].values, y_test['LATITUDE'].values
    coords_error = sqrt(square(x - x0) + square(y - y0))
    
    standard_error = (building_penalty * build_missclass + floor_penalty *
                      floor_missclass + sum(coords_error))
    # error probability for error being less than 10 meters
    coords_error_prob = (coords_error[coords_error < 10].shape[0] / 
                         coords_error.shape[0] * 100)
    
    knn_errors = (build_missclass, floor_missclass, coords_error, standard_error, 
              coords_error_prob)
    print("position error:")
    
    mean_c = mean(coords_error)
    std_c = std(coords_error)
    
    build_error = build_missclass / y_test.shape[0] * 100 # Percent Error
    floor_error = floor_missclass / y_test.shape[0] * 100 # Percent Error
    
    str1 = "Totals Output:"
    str2 = "Mean Coordinate Error: %.2f +/- %.2f meters" % (mean_c, std_c)
    str3 = "Standard Error: %.2f meters" % standard_error
    str4 = "Building Percent Error: %.2f%%" % build_error
    str5 = "Floor Percent Error: %.2f%%" % floor_error
    
    if coords_error_prob != "N/A":
        str6 = "Prob that Coordinate Error Less than 10m: %.2f%%" %coords_error_prob    
    else:
        str6 = ""
    
    s_report = '\n'.join([str1, str2, str3, str4, str5, str6])

    print(s_report)




    toc = time() #program end time
    print("Program Timer: %.2f seconds" % (toc-tic))
    

    
############################
# Representation of results
############################

# Put the result into a color plot
plt.figure()
plt.scatter(x0, y0, alpha=0.5, s=7, marker='o', label='actual position')
plt.scatter(x, y, alpha=0.3, s=4, marker='o', label='predicted position')
plt.legend()
plt.grid()
plt.xlabel('Longitude (m)', fontweight ='bold')
plt.ylabel('Latitude (m)', fontweight ='bold')
plt.title("Data points", fontweight ='bold',fontsize =20)
plt.tight_layout(pad=0)
fig = plt.figure(figsize=(100,100))
plt.savefig('data_points.png')
plt.show()

# Creating figure
fig = plt.figure(figsize = (10, 10))
ax = plt.axes(projection ="3d")
ax.view_init(15, 35)

# Creating plot
l1 = ax.scatter3D(x, y, prediction["FLOOR"].values, color = "darkorange" , marker = '*')
ax.set_xlabel('LONGITUDE', fontweight ='bold', fontsize =12)
ax.set_ylabel('LATITUDE', fontweight ='bold', fontsize =12)
ax.set_zlabel('FLOOR', fontweight ='bold', fontsize =12)

l2 = ax.scatter3D(x0,  y0, y_test["FLOOR"].values, color = "darkcyan" , marker = 'x')
plt.title("actual and predicted positions", fontweight ='bold',fontsize =20)
ax.set_xlabel('LONGITUDE (m)', fontweight ='bold', fontsize =12)
ax.set_ylabel('LATITUDE (m)', fontweight ='bold', fontsize =12)
ax.set_zlabel('FLOOR', fontweight ='bold', fontsize =12)

ax.legend([l2,l1], ['actual position', 'predicted position'], numpoints = 1)
# show plot
plt.show()

# Save outputs into a CSV file

output = DataFrame({'actual_longitude (m)': x0 , 
                       'actual_latitude (m)': y0, 
                       'predicted_longitude (m)': x, 
                       'predicted_latitude (m)': y ,
                       'error_longitude (m)': (x0 - x),
                       'error_latitude (m)':  (y0 - y),
                       'distance_error(m)': coords_error})


with open('knn_output.csv', 'w') as f:
    output.to_csv(f,header = True, index = False)
    
output



