***************************************************************
***************************************************************
TV Human Interaction Dataset

The dataset was compiled from over 20 different TV shows and consist of
300 video clips containing 4 interactions: hand shakes, high fives, hugs 
and kisses, as well as clips that don't contain any of the interactions. This
dataset was used in:

	High Five: Recognising human interactions in TV shows
	Patron-Perez, A., Marszalek, M., Zisserman, A. and Reid, I.
	In Proc.  BMVC, Aberystwyth, UK, 2010 

In the paper the dataset was split in two mutually exclusive sets for testing and
training:

SET 1:
	Hand shake 		-> {2,14,15,16,18,19,20,21,24,25,26,27,28,32,40,41,42,43,44,45,46,47,48,49,50}
	High five		-> {1,6,7,8,9,10,11,12,13,23,24,25,27,28,29,30,31,32,33,34,35,44,45,47,48}
	Hug			-> {2,3,4,11,12,15,16,17,18,20,21,27,29,30,31,32,33,34,35,36,42,44,46,49,50}	
	Kiss			-> {1,7,8,9,10,11,12,13,14,16,17,18,22,23,24,26,29,31,35,36,38,39,40,41,42}
	Negative		-> 1-50

SET 2:
	Hand shake 		-> {1,3,4,5,6,7,8,9,10,11,12,13,17,22,23,29,30,31,33,34,35,36,37,38,39}
	High five		-> {2,3,4,5,14,15,16,17,18,19,20,21,22,26,36,37,38,39,40,41,42,43,46,49,50}
	Hug			-> {1,5,6,7,8,9,10,13,14,19,22,23,24,25,26,28,37,38,39,40,41,43,45,47,48}
	Kiss			-> {2,3,4,5,6,15,19,20,21,25,27,28,30,32,33,34,37,43,44,45,46,47,48,49,50}
	Negative		-> 51-100

***************************************************************
***************************************************************
Annotation files description:

-----------------------------------------------------------------------------------------
 <interaction>_<idx>.annotations

	+ First line:
		#num_frames: <n>     
	
		<n> :=  number of annotated frames  in the video.

	+ Rest of the file is structured by frames

		#frame: <f>  #num_bbxs: <d> [ #interactions: < id_i - id_j> ]
		<id_1>  <bbx_1>  <label_1>  <ho_1>
		.
		.
		<id_d>  <bbx_d>  <label_d>  <ho_d>

		<f> 			:=  frame number
		<d>			:=  number of upper body annotations in the frame
		<id_i - id_j> :=  id's of people  interacting in this frame (if any)
		<id_i>		:=  person ID corresponding to the i-th bounding box in this frame. 
		<bbx_i>	:=  bounding box dimensions in pixel coordinates: [top_left_x  top_left_y size]
		<label_i> 	:=  interaction label of  i-th annotation		
		<ho_i>		:=  discrete head orientation of i-th annotation 

-----------------------------------------------------------------------------------------

	













