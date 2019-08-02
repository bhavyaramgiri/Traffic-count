# the code is divided into 6 parts 3 common parts and 3 seperate parts for vehicles going straight
# and vehicles joining the straight road from a turn.
#--------------------------------------------------------------------------------------------------
# 1st part

import numpy as np
import cv2
import pandas as pd

fgbg = cv2.createBackgroundSubtractorMOG2()# create background subtractor
cap = cv2.VideoCapture('traffic4.mp4') # captures the video
fps = cap.get(cv2.CAP_PROP_FPS) # frames per second
fps = int(fps)
frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # total frame count
frames_count = int(frames_count)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # Width of the frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # height of the frame

print("Number of frames in entire video :{}".format(frames_count))
print("Number of frames per second:{}".format(fps))
print("Width of the video:{}".format(int(width)))     
print("Height of the video:{}".format(int(height))) 
  
# we will require dataframe once we are inside the while loop to store the centroids of the vehicles
# one data frame for the cars continuing going straight and other dataframe for the cars joing the 
# straight road from the turn.
df = pd.DataFrame(index=range(int(frames_count)))
df.index.name = "Frames"
df_turn = pd.DataFrame(index=range(int(frames_count)))
df_turn.index.name = "Frames"

# all the variables below will be updated frame by frame once inside the while loop.
framenumber = 0  # frame number will be updated for each loop
carscrossed_straight = 0  
carscrossed_turn = 0 
carids = [] # vehicle unique id are appended in the empty list which enters the frame 
carids_turn = []
caridscrossed = [] # vehicles which cross the green line are noted down
caridscrossed_turn = []
totalcars = 0 # Total number of the cars as per the ids detected will be updated inside the loop.
totalcars_turn = 0

# To capture the changes made on the video, creating "video" object
ret, frame = cap.read()  # import image
h = int(frame.shape[0])
w = int(frame.shape[1])
height = int(h/2)
width = int(w/2) 
dsize = (width, height)
image = cv2.resize(frame,dsize)  # resize image
width2, height2, channels = image.shape
# Above codes are for finding fps, height and width of the frame which are the inputs to  cv2.VideoWriter() function.
video = cv2.VideoWriter('traffic_count.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2))


while True: # while loop will execute the code below as long as check returns True.
    ret, frame = cap.read() # check/ret have same function of storing boolean values
    if ret: # if ret is True the following code will execute or else it will break
        
        #reducing every frame by 50% 
        scale_percent = 50 
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dsize = (width, height)
        image = cv2.resize(frame,dsize)  # resize image
        
        # processing every frame to draw contours
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
        fgmask = fgbg.apply(gray) # applying background subtractor
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)) # initializing ellipse kernel 
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(opening, kernel)
        ret, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY) # converson to binary
        contours, hierarchy = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # fining contours
        hull = [cv2.convexHull(c) for c in contours] # creating hull around contour points 
        cv2.drawContours(image, hull, -1, (255,255,255), 2) # drawing contour
        
        # drawing the lines for carscrossed_straight
        # all the ids are considered for the vehicles below blue line
        # all the vehicles which cross green line will be counted
        blue_x_line1a = 290
        blue_y_line1a = 190
        blue_x_line1b = 540
        blue_y_line1b = 260
        cv2.line(image, (blue_x_line1a, blue_y_line1a), (blue_x_line1b, blue_y_line1b), (255,0,0), 2) 
        green_x_line1a = 270
        green_y_line1a = 220
        green_x_line1b = 510
        green_y_line1b = 290
        cv2.line(image, (green_x_line1a, green_y_line1a), (green_x_line1b, green_y_line1b), (0, 255, 0), 2)
        # drawing the line for carcrossed_turn 
        blue_x_line2a = 750
        blue_y_line2a = 480
        blue_x_line2b = 750
        blue_y_line2b = 180
        cv2.line(image, (blue_x_line2a,blue_y_line2a), (blue_x_line2b, blue_y_line2b), (255, 0, 0), 2)
        green_x_line2a = 795
        green_y_line2a = 490
        green_x_line2b = 795
        green_y_line2b = 200
        cv2.line(image, (green_x_line2a,green_y_line2a), (green_x_line2b,green_y_line2b), (0,255,0), 2)
# -------------------------------------------------------------------------------------------------
# 2nd part
       # criterias
        minarea = 7000 # minimum area of the contours allowed
        maxarea = 60000 # maximum area of the contour allowed
        cxx = [] # The x co-ordinate of centroid will be updated
        cyy = [] # the y co-ordinate of the centroid will be updated

        for i in range(len(contours)): # of all the contours in the frame
            if hierarchy[0, i, 3] == -1: # choosing only the parent ones
                area = cv2.contourArea(contours[i]) # area of the parent contour is identified
                if minarea < area < maxarea: # the area is checked if is in between the contraints
                    cnt = contours[i] 
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00']) # x co-ordinate of centroid
                    cy = int(M['m01'] / M['m00']) # y co-ordinate of centroid
                    # by adding markers, it can be verified that the contours are drawn for one vehicles only. 
                    if cy > blue_y_line1a and cx < blue_x_line1b:
                        cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=10,line_type=cv2.LINE_AA)
                        cxx.append(cx)
                        cyy.append(cy)
                        
                        
        # same process is applied for vehicle joing from the turn to the straight road                
        minarea_turn = 1000 
        maxarea_turn = 40000 
        cxx_turn = []
        cyy_turn = []
        
        for i in range(len(contours)):
            if hierarchy[0, i, 3] == -1:
                area = cv2.contourArea(contours[i])
                if minarea_turn < area < maxarea_turn:
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    # by adding markers, it can be verified that the contours are drawn for one vehicles only. 
                    if cx >= blue_x_line2a:
                        cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=10,line_type=cv2.LINE_AA)
                        cxx_turn.append(cx)
                        cyy_turn.append(cy)
#--------------------------------------------------------------------------------------------------       
# 3rd part for vehicles continuing straight

        min_index = []
        maxrad = 70
        # this will only execute for the frames with contours and centroids
        if len(cxx):  # if cxx is filled and not empty
            if not carids: # if there are no vehicles "ids" recorded yet
                for i in range(len(cxx)):  
                    carids.append(i)  
                    df[str(carids[i])] = "" # creating an empty column of current carid
                    df.at[int(framenumber), str(carids[i])] = [cxx[i], cyy[i]] # adding the centroids inside the df
                    totalcars = carids[i] + 1  # updating the total cars recognized.
 
            else:  # if there are already car ids in the list
                dx = np.zeros((len(cxx), len(carids)))  # new arrays to calculate the difference between oldcent and current
                dy = np.zeros((len(cyy), len(carids)))  
 
                for i in range(len(cxx)):  
                    for j in range(len(carids)):  # loops through all recorded car ids
                        oldcxcy = df.iloc[int(framenumber - 1)][str(carids[j])]# acquires centroid from previous frame for specific carid
                        curcxcy = [cxx[i], cyy[i]]# acquires current frame centroid that doesn't necessarily line up with previous frame centroid
                        if oldcxcy == curcxcy: # if older and current centroid is same, then they are the same vehicle. 
                            df.at[int(framenumber), str(carids[j])] = curcxcy
                            min_index.append(i)
                            
                        if not oldcxcy:  # checks if old centroid is empty 
                            continue  # continue to next carid
                        else:  # difference between all current centroids of this frame and old centroids of previous are stored.
                            dx[i, j] = np.abs(oldcxcy[0] - curcxcy[0])
                            dy[i, j] = np.abs(oldcxcy[1] - curcxcy[1])
 
                for j in range(len(carids)):  
                    sumsum = dx[:, j] + dy[:, j] # array of difference of a particular column
                    correctindextrue = np.argmin(sumsum) # choosing the minimum value from array above and storing index number
                    
 
                    # The index is used to extract the minimum difference in order to check if it is within radius later on
                    mindx = dx[correctindextrue, j]
                    mindy = dy[correctindextrue, j]
                     
                    if mindx == 0 and mindy == 0: # this is for not considering the unfilled dx and dy elements
                        continue
                    else:
                        if mindx < maxrad and mindy < maxrad: # if the difference is less than the maximum radius than its the same car
                           #adds centroid to corresponding previously existing carid
                            df.at[int(framenumber), str(carids[j])] = [cxx[correctindextrue], cyy[correctindextrue]]
                            min_index.append(correctindextrue)
                
        
                for i in range(len(cxx)): 
                    if i not in min_index:
                        df[str(totalcars)] = ""  # create another column with total cars
                        carids.append(totalcars)  # append to list of car ids
                        df.at[int(framenumber), str(totalcars)] = [cxx[i], cyy[i]]  # add centroid to the new car id
                        totalcars = totalcars + 1  # adds another total car the count
                        
#-------------------------------------------------------------------------------------------                
# 3rd part for vehicles joining the straight road from the turn.
# The code below is same as above with change in variables names
        min_index_turn = []
        maxrad_turn = 70
        
        if len(cxx_turn):   
            if not carids_turn: 
                for i in range(len(cxx_turn)):  
                    carids_turn.append(i)  
                    df_turn[str(carids_turn[i])] = "" 
                    df_turn.at[int(framenumber), str(carids_turn[i])] = [cxx_turn[i], cyy_turn[i]]
                    totalcars_turn = carids_turn[i] + 1  
 
            else:  
                dx_turn = np.zeros((len(cxx_turn), len(carids_turn)))  
                dy_turn = np.zeros((len(cyy_turn), len(carids_turn)))  
 
                for i in range(len(cxx_turn)):  
                    for j in range(len(carids_turn)):  
                        oldcxcy_turn = df_turn.iloc[int(framenumber - 1)][str(carids_turn[j])]
                        curcxcy_turn = [cxx_turn[i], cyy_turn[i]]
                        if oldcxcy_turn == curcxcy_turn:
                            df_turn.at[int(framenumber), str(carids_turn[j])] = curcxcy_turn
                            min_index_turn.append(i)
                            
                        if not oldcxcy_turn:  
                            continue  
                        else:  
                            dx_turn[i, j] = np.abs(oldcxcy_turn[0] - curcxcy_turn[0])
                            dy_turn[i, j] = np.abs(oldcxcy_turn[1] - curcxcy_turn[1])
 
                for j in range(len(carids_turn)):  
                    sumsum_turn = dx_turn[:, j] + dy_turn[:, j]  
                    correctindextrue_turn = np.argmin(sumsum_turn)
                    
                    mindx_turn = dx_turn[correctindextrue_turn, j]
                    mindy_turn = dy_turn[correctindextrue_turn, j]
            
                     
                    if mindx_turn == 0 and mindy_turn == 0: 
                        continue
                    else:
                        if mindx_turn < maxrad_turn and mindy_turn < maxrad_turn:
                           #adds centroid to corresponding previously existing carid
                            df_turn.at[int(framenumber), str(carids_turn[j])] = [cxx_turn[correctindextrue_turn], cyy_turn[correctindextrue_turn]]
        
                            min_index_turn.append(correctindextrue_turn)
                
        
                for i in range(len(cxx_turn)): 
                    if i not in min_index_turn:
                        df_turn[str(totalcars_turn)] = "" 
                        carids_turn.append(totalcars_turn)  
                        df_turn.at[int(framenumber), str(totalcars_turn)] = [cxx_turn[i], cyy_turn[i]]  
                        totalcars_turn = totalcars_turn + 1 
                        
#--------------------------------------------------------------------------------------------
# 4th part for vehicles continuing straight        
               
        currentcars = 0  # current cars on screen
        currentcarsindex = []  # current cars on screen carid index


        for i in range(len(carids)):  # loops through all carids                     
                        
            if df.at[int(framenumber), str(carids[i])] != '':
            
                currentcars = currentcars + 1  # adds another to current cars on screen
                currentcarsindex.append(i)  # adds car ids to current cars on screen       
                
#---------------------------------------------------------------------------------------------                
# 4th part for vehicles joining the straight road from the turn. 
                       
        currentcars_turn = 0  # current cars on screen
        currentcarsindex_turn = []  # current cars on screen carid index


        for i in range(len(carids_turn)):  # loops through all carids                     
                        
            if df_turn.at[int(framenumber), str(carids_turn[i])] != '':
            
                currentcars_turn = currentcars_turn + 1  # adds another to current cars on screen
                currentcarsindex_turn.append(i)  # adds car ids to current cars on screen

#--------------------------------------------------------------------------------------------
# 5th part for vehicles continuing straight                        
        for i in range(currentcars):
            curcent = df.iloc[int(framenumber)][str(carids[currentcarsindex[i]])]
            
            if framenumber != 0: # This is to avoid for choosing the last frame from the df if the frame is 0
                oldcent = df.iloc[int(framenumber - 1)][str(carids[currentcarsindex[i]])]
            else:
                oldcent = []
 
            if curcent:             
                cv2.putText(image, "ID:" + str(carids[currentcarsindex[i]]), (int(curcent[0]), int(curcent[1] - 10)),cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)           
             
            
                if oldcent:  # checks if old centroid exists
                # the circle below is visualization for area within which curcent and oldcent are of the same vehicle.
                    cv2.circle(image,(tuple(oldcent)), maxrad, (0,0,255), 1)
                    
                
                # checks if old centroid is on or below line and curcent is on or above line
                    if oldcent[0] <= green_x_line1b and oldcent[1] >= green_y_line1a and curcent[0] <= green_x_line1b and curcent[1] < green_y_line1a and carids[currentcarsindex[i]] not in caridscrossed:
                        carscrossed_straight = carscrossed_straight + 1
                        cv2.line(image,(green_x_line1a, green_y_line1a), (green_x_line1b, green_y_line1b), (0, 0, 255), 5)
                        caridscrossed.append(currentcarsindex[i])  # adds car id to list of count cars to prevent double counting

#---------------------------------------------------------------------------------------------
# 5th part for vehicles joining the straight road from the turn. 
        for i in range(currentcars_turn):  # loops through all current car ids on current frame
            curcent_turn = df_turn.iloc[int(framenumber)][str(carids_turn[currentcarsindex_turn[i]])]
            
            if framenumber != 0:
                oldcent_turn = df_turn.iloc[int(framenumber - 1)][str(carids_turn[currentcarsindex_turn[i]])]
            else:
                oldcent_turn = []
 
            if curcent_turn:  # if there is a current centroid            
                cv2.putText(image, "ID:" + str(carids_turn[currentcarsindex_turn[i]]), (int(curcent_turn[0]), int(curcent_turn[1] - 10)),cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)                    
            
                if oldcent_turn:  # checks if old centroid exists
                # the circle below is visualization for area within which curcent and oldcent are of the same vehicle.
                    cv2.circle(image,(tuple(oldcent_turn)), maxrad_turn, (0,0,255), 1)
                
                # checks if old centroid is on or below line and curcent is on or above line
                    if oldcent_turn[0] >= green_x_line2a and curcent_turn[0] <= green_x_line2a and carids_turn[currentcarsindex_turn[i]] not in caridscrossed_turn:
                        carscrossed_turn = carscrossed_turn + 1
                        cv2.line(image, (green_x_line2a,green_y_line2a), (green_x_line2b,green_y_line2b), (0, 0, 255), 5)
                        caridscrossed_turn.append(currentcarsindex_turn[i])
#----------------------------------------------------------------------------------------------
# 6th part    
    # Top left hand corner on-screen text
        cv2.rectangle(image, (0, 0), (250, 100), (169,169,169), -1)  # background rectangle for on-screen text
 
        cv2.putText(image, "Cars Crossed straight: " + str(carscrossed_straight), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0),2)
        cv2.putText(image, "Cars Crossed turn: " + str(carscrossed_turn), (0, 45), cv2.FONT_HERSHEY_SIMPLEX, .5,(0, 0, 0), 2)
 
        cv2.putText(image, "Total Cars Detected: " + str(len(carids) + len(carids_turn)), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, .5,(0, 0, 0), 2)
 
        cv2.putText(image, "Frame: " + str(framenumber) + ' of ' + str(frames_count), (0, 75), cv2.FONT_HERSHEY_SIMPLEX,.5, (0, 0, 0), 2)
 
        cv2.putText(image, 'Time: ' + str(round(framenumber / fps, 2)) + ' sec of ' + str(round(frames_count / fps, 2))+ ' sec', (0, 90), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)
    
        cv2.imshow("contour_image", image)
    
        framenumber = framenumber + 1
        
        # All the changes made to the video will be written and stored to the video itself
        video.write(image)
    
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            break
    # if ret inside the while loop of 1st part returned False then the code would break here.       
    else:
        
        break
        
# saves dataframe to csv file for later analysis
df.to_csv('traffic.csv', sep=',')        
df_turn.to_csv('traffic_turn.csv', sep=",")
cap.release()
cv2.destroyAllWindows()