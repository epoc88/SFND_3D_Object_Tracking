# SFND : Camera Based 3D Object Tracking Report

-------------

- ####  FP.1 Match 3D Objects
	- Task:  Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.
	- Implementation : 

```
	void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
	{
	    int curBoxCount = currFrame.boundingBoxes.size();
	    int preBoxCount = prevFrame.boundingBoxes.size();
	    int ptBoxCounts [curBoxCount][preBoxCount] = {};

	    // Iterate over current and previous frame to find bounding boxes contains matches
	    for (const cv::DMatch& match : matches)
	    {
		for (const BoundingBox& bbCur : currFrame.boundingBoxes)
		{
		    for (const BoundingBox& bbPre : prevFrame.boundingBoxes)
		    {
		        if (bbCur.roi.contains(currFrame.keypoints[match.trainIdx].pt) &&
		            bbPre.roi.contains(prevFrame.keypoints[match.queryIdx].pt))
		        {
		            ptBoxCounts[bbCur.boxID][bbPre.boxID] += 1;
		        }
		    }
		}
	    }


	    // Find the highest number of counts from previous frame for each box in the current frame
	    for (const BoundingBox& bbCur : currFrame.boundingBoxes)
	    {
		int maxCount = 0;
		int maxId = 0;

		for (const BoundingBox& bbPre : prevFrame.boundingBoxes)
		{
		    if (maxCount < ptBoxCounts[bbCur.boxID][bbPre.boxID])
		    {
		        maxCount = ptBoxCounts[bbCur.boxID][bbPre.boxID];
		        maxId = bbPre.boxID;
		    }
		}
		bbBestMatches.insert (std::pair<int,int>(maxId,bbCur.boxID));
	    }

	}

```

- ####  FP.2 Compute Lidar-based TTC
 	- Task:  Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.
	- Implementation :

    Remove effects of outliers: firstly, considering only lidar points within the ego lane for TTC calculation. secondly, the average distances are considered for previous and current lidar points and used in the TTC calculation formula. 
```
	void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
		             std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
	{
	    vector<double> prevXPoints, currXPoints;
	    double egoLaneWidth = 2.92;

	    for (auto it = lidarPointsPrev.begin(); it!=lidarPointsPrev.end();it++)
	    {
		if(abs(it->y) <= egoLaneWidth/2.0)
		    prevXPoints.push_back(it->x);
	    }

	    for (auto it = lidarPointsCurr.begin(); it!=lidarPointsCurr.end();it++)
	    {
		if(abs(it->y) <= egoLaneWidth/2.0)
		    currXPoints.push_back(it->x);
	    }

	    double minXPrev = 0 , minXCurr = 0 ;

	    if(prevXPoints.size() > 0 && currXPoints.size()>0)
	    {
		for(auto x : prevXPoints)
		    minXPrev += x ;
		minXPrev = minXPrev/prevXPoints.size();
		for(auto y : currXPoints)
		    minXCurr += y ;
		minXCurr = minXCurr/currXPoints.size();
	    }

	    // calculate time to collision
	    const double dT = 1.0 / frameRate;
	    TTC = minXCurr * dT / (minXPrev - minXCurr);

	}

```
 - ####  FP.3 Associate Keypoint Correspondences with Bounding Boxes
 	- Task : Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.
 	- Implementation  : 
 	 Iterating over the keypoint matches and calculating the euclidean distance if the points located within the bounding box ROI. AverageDistance over all the calculated euclidean points is calculated. Then iterating over the keypoint matches , those matches whose points fall into the bounding box ROI are added to the bounding box kptMatches, as long as their distances are within threshold(2.5 has been set as variation between frames are small).

```
	void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
	{
	
	    // Find the matches within current and previous boundingBox
	    std::vector<double> kptMatchesInROI ;
	    for(auto it = kptMatches.begin();it!= kptMatches.end();it++)
	    {
		cv::KeyPoint curKeyPt = kptsCurr[it->trainIdx];
		cv::KeyPoint preKeyPt = kptsPrev[it->queryIdx];
		

		if(boundingBox.roi.contains(curKeyPt.pt))
		{
		    kptMatchesInROI.push_back(cv::norm(curKeyPt.pt-preKeyPt.pt));
		}

	    }

	    //calculatig the average kptMatches within boundingBox
	    double averageDist = std::accumulate(kptMatchesInROI.begin(),kptMatchesInROI.end(),0.0) / kptMatchesInROI.size();

	    //pushback kptMatches for boundingBox within the threshold
	    double threshold= averageDist+2.5;

	    for(auto it = kptMatches.begin();it!= kptMatches.end();it++)
	    {
		cv::KeyPoint curKeyPt = kptsCurr[it->trainIdx];
		cv::KeyPoint preKeyPt = kptsPrev[it->queryIdx];
		

		if(boundingBox.roi.contains(curKeyPt.pt))
		{
		    double kptdist = cv::norm(curKeyPt.pt-preKeyPt.pt);

		    if(kptdist < threshold)
		    {
		        boundingBox.kptMatches.push_back(*it);
		    }
		}
	    }
	}
```

 - ####  FP.4 Compute Camera-based TTC
 	- Task : Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame
  	- Implementation :
  	The distance is computed by the keypoint correspondences matches from the previous and current frame. To get a reliable value for the size variation of the preceding vehicle, median value distance measurements are used for computing keypoints. 

     ```
	void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
		              std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
	{


	    // compute distance ratios between all matched keypoints
	    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. image
	    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
	    { // outer kpt. loop

		// get current keypoint and its matched partner in the prev.image
		cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
		cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

		for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
		{ // inner kpt.-loop

		    double minDist = 100.0; // min. required distance

		    // get next keypoint and its matched partner in the previous image
		    cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
		    cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

		    // compute distances and distance ratios
		    double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
		    double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

		    if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
		    { // avoid division by zero
		        double distRatio = distCurr / distPrev;
		        distRatios.push_back(distRatio);
		    }
		}
	    }

	    // only continue if list of distance ratios is not empty
	    if (distRatios.size() == 0)
	    {
		TTC = NAN;
		return;
	    }

	    std::sort(distRatios.begin(), distRatios.end());

	    double medianDistRatio;
	    if (distRatios.size()%2==1)
	    {
		medianDistRatio = distRatios[distRatios.size()/2];
	    }
	    else
	    {
		int index = distRatios.size()/2;
		medianDistRatio = (distRatios[index-1] + distRatios[index])/2;
	    }

	    TTC = - 1 / (frameRate * (1 - medianDistRatio));      //  -dT/(1 - medianDistRatio)
	}
```
```

 - ####  FP.5 Performance Evaluation 1	
 	- Task : Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.
  	- Evaluation :
	

   Some examples of not plausable cases :
 
  	<p align="center">
	<img src= "https://github.com/epoc88/SensorFusion_3DObjectTracking/blob/master/docs/3.png" width=900 height=300>
	<img src= "https://github.com/epoc88/SensorFusion_3DObjectTracking/blob/master/docs/13.png" width=900 height=300>
	<img src= "https://github.com/epoc88/SensorFusion_3DObjectTracking/blob/master/docs/14.png" width=900 height=300>
    </p>
    
   
* In task2 some outliers has been eliminated when computing TTC, however, it has not been completely removed. There are still outliers as which cause inaccurate TTC. like above 3 scenarios. 
* When the front vechicles slowing down. According to the TTC Lidar calculation formula, smaller moving distance may cause higher change in TTC value. Thus leading to the implausible TTC measurement.
	

  
 - #### FP.6 Performance Evaluation 2	
 	- Task : Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.
  	- Evaluation :
 
    <p align="center">
    </p> 

 * TOP3 detector / descriptor combinations are chosen based on the achieve minimal processing time with significant matches 
 	Best combinations : 
    	`FAST + BRIEF`,
       	`FAST + BRISK`,
	`FAST + ORB`

        

 * As it can be seen from the graphs, the camera TTC calculation is way off at some frames. Keypoint mismatch between frames and slight shift of keypoints in next frame is observed in matching. This results to inaccurate calculation of TTC.
 * The model assumed for the TTC calculation is constant velocity model. However in the real scenario the preceding vehicle accelerates/decelerates and this affects the accuracy of the TTC calculation.


-------------
	



