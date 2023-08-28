# SEESZooniverse

**This is the GitHub repository for the following research project:**
_Uniting Machine Learning and Citizen Science for Automatic Land Cover Classification_

**Authors:**
Oseremen Ojiefoh, Roayba Adhi, Riya Tyagi, Lavanya Gnanakumar, Haitham Ahmad, Kavya Ram, Rusty Low, Peder Nelson

**Abstract:**
To complement satellite land cover data, researchers today rely on citizen science to record and analyze ground photos of land cover because of its ability to efficiently facilitate research at a large scale, quickly gather data across a variety of regions, and ethically engage communities in pertinent matters. The GLOBE Observer App, part of The GLOBE Program, is a citizen science tool used to document the planet. To make a land cover observation, a user logs their date and location, records the surface conditions, and takes six directional photos. The land cover features present are then identified in each photo, with users dragging sliders to report the percentage of land or sky that each feature covers in the image. However, our surveys have found the classification step can be a complicated process. User subjectivity and limited knowledge of land cover terminology, paired with the time-consuming nature of classification, caused 46% of survey-takers to report that they often or always skip the step of classifying altogether, substantially decreasing data usability for scientists. In this study, we contributed to developing a geographically diverse dataset of 5,896 directional land cover photos, which were then uploaded to the GLOBE database. Using Zooniverse, a citizen science annotation tool, we collected labels determining whether the three classes—sky, land, or water—were present in each land cover photo. Three Resnet-18 machine learning models were trained to classify sky, land, and water in each photo, treating Zooniverse labels as ground truth. The sky, land, and water models received an accuracy of 94%, 92%, and 95%, respectively. We then used the trained models for feature extraction. Features were passed into three Support Vector Machines, which determined the percentages of sky, land, and water in each photo. Our model pipeline can assist untrained volunteers with future image classifications, resulting in a higher quality dataset for scientists.

**Acknowledgements**
These data were obtained from NASA and the GLOBE Program and are freely available for use in research, publications and commercial applications.

The authors would like to acknowledge the support of the 2022 Earth System Explorers Team, NASA STEM Enhancement in the Earth Sciences (SEES) Virtual High School Internship program. The NASA Earth Science Education Collaborative leads Earth System Explorers through an award to the Institute for Global Environmental Strategies, Arlington, VA (NASA Award NNX6AE28A).
The SEES High School Summer Intern Program is led by the Texas Space Grant Consortium at the University of Texas at Austin (NASA Award NNX16AB89A).
