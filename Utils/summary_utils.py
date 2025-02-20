import numpy as np
import cv2



# Code adapted from: https://github.com/e-apostolidis/PGL-SUM/blob/master/evaluation/generate_summary.py
def generate_summary_single(shot_bound:list,score:np.ndarray,n_frames:int,positions:int,return_shot_info:bool = False):
    """ Input:
        shot_bound: the shot boundaries as a list of lists
        score : frame-wise relevance scores of the video
        n_frames: number of frames in the video
        positions: the indices 
        return_shot_info: Return details of shots if true
        Output:
        shot_lengths: length of each shot
        shot_imp_scores: The importance scores assigned to each shot
        summary: 0/1 array of all selected frames

    """
    frame_init_scores = score
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])

    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i + 1]
        if i == len(frame_init_scores):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = frame_init_scores[i]

    # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
    shot_imp_scores = []
    shot_lengths = []
    for shot in shot_bound:
        shot_lengths.append(shot[1] - shot[0] + 1)
        shot_imp_scores.append((frame_scores[shot[0]:shot[1] + 1].mean()).item())

    # Select the best shots using the knapsack implementation
    final_shot = shot_bound[-1]
    final_max_length = int((final_shot[1] + 1) * 0.15)

    selected = knapSack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))

    # Select all frames from each selected shot (by setting their value in the summary vector to 1)
    summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
    for shot in selected:
        summary[shot_bound[shot][0]:shot_bound[shot][1] + 1] = 1
    if return_shot_info:
        return shot_lengths, shot_imp_scores,selected,summary
    return summary



# A modified evaluate correlation to pick out indices from a location 


# link: https://github.com/wulfebw/algorithms/blob/master/scripts/dynamic_programming/knapsack.py
          
     
      
def knapSack(W, wt, val, n):
	""" Maximize the value that a knapsack of capacity W can hold. You can either put the item or discard it, there is
	no concept of putting some part of item in the knapsack.

	:param int W: Maximum capacity -in frames- of the knapsack.
	:param list[int] wt: The weights (lengths -in frames-) of each video shot.
	:param list[float] val: The values (importance scores) of each video shot.
	:param int n: The number of the shots.
	:return: A list containing the indices of the selected shots.
	"""
	K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

	# Build table K[][] in bottom up manner
	for i in range(n + 1):
		for w in range(W + 1):
			if i == 0 or w == 0:
				K[i][w] = 0
			elif wt[i - 1] <= w:
				K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
			else:
				K[i][w] = K[i - 1][w]

	selected = []
	w = W
	for i in range(n, 0, -1):
		if K[i][w] != K[i - 1][w]:
			selected.insert(0, i - 1)
			w -= wt[i - 1]

	return selected


# Shot boundary F1 score, based off of code source : https://stackoverflow.com/questions/64860091/computing-macro-average-f1-score-using-numpy-pythonwithout-using-scikit-learn

def evaluate_f1_summaries(summary_1,summary_2):
    overlap = summary_2 & summary_1
    precision = overlap.sum()/summary_1.sum()
    recall = overlap.sum()/summary_2.sum()
    return 2*precision*recall/(precision+recall)


def calculate_metrics(true_boundaries, predicted_boundaries):
    TP = len(set(true_boundaries) & set(predicted_boundaries))
    FP = len(set(predicted_boundaries) - set(true_boundaries))
    FN = len(set(true_boundaries) - set(predicted_boundaries))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
  

    return precision, recall, f1_score



def evaluate_scenes(gt_scenes, pred_scenes, return_mistakes=False, n_frames_miss_tolerance=2):
    """
    Adapted from: https://github.com/gyglim/shot-detection-evaluation
    The original based on: http://imagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=19

    n_frames_miss_tolerance:
        Number of frames it is possible to miss ground truth by, and still being counted as a correct detection.

    Examples of computation with different tolerance margin:
    n_frames_miss_tolerance = 0
      pred_scenes: [[0, 5], [6, 9]] -> pred_trans: [[5.5, 5.5]]
      gt_scenes:   [[0, 5], [6, 9]] -> gt_trans:   [[5.5, 5.5]] -> HIT
      gt_scenes:   [[0, 4], [5, 9]] -> gt_trans:   [[4.5, 4.5]] -> MISS
    n_frames_miss_tolerance = 1
      pred_scenes: [[0, 5], [6, 9]] -> pred_trans: [[5.0, 6.0]]
      gt_scenes:   [[0, 5], [6, 9]] -> gt_trans:   [[5.0, 6.0]] -> HIT
      gt_scenes:   [[0, 4], [5, 9]] -> gt_trans:   [[4.0, 5.0]] -> HIT
      gt_scenes:   [[0, 3], [4, 9]] -> gt_trans:   [[3.0, 4.0]] -> MISS
    n_frames_miss_tolerance = 2
      pred_scenes: [[0, 5], [6, 9]] -> pred_trans: [[4.5, 6.5]]
      gt_scenes:   [[0, 5], [6, 9]] -> gt_trans:   [[4.5, 6.5]] -> HIT
      gt_scenes:   [[0, 4], [5, 9]] -> gt_trans:   [[3.5, 5.5]] -> HIT
      gt_scenes:   [[0, 3], [4, 9]] -> gt_trans:   [[2.5, 4.5]] -> HIT
      gt_scenes:   [[0, 2], [3, 9]] -> gt_trans:   [[1.5, 3.5]] -> MISS
    """

    shift = n_frames_miss_tolerance / 2
    gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])

    gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1)
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1)

    i, j = 0, 0
    tp, fp, fn = 0, 0, 0
    fp_mistakes, fn_mistakes = [], []

    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans):
            fn += 1
            fn_mistakes.append(gt_trans[i])
            i += 1
        elif i == len(gt_trans):
            fp += 1
            fp_mistakes.append(pred_trans[j])
            j += 1
        elif pred_trans[j, 1] < gt_trans[i, 0]:
            fp += 1
            fp_mistakes.append(pred_trans[j])
            j += 1
        elif pred_trans[j, 0] > gt_trans[i, 1]:
            fn += 1
            fn_mistakes.append(gt_trans[i])
            i += 1
        else:
            i += 1
            j += 1
            tp += 1

    if tp + fp != 0:
        p = tp / (tp + fp)
    else:
        p = 0

    if tp + fn != 0:
        r = tp / (tp + fn)
    else:
        r = 0

    if p + r != 0:
        f1 = (p * r * 2) / (p + r)
    else:
        f1 = 0

    assert tp + fn == len(gt_trans)
    assert tp + fp == len(pred_trans)

    if return_mistakes:
        return p, r, f1, (tp, fp, fn), fp_mistakes, fn_mistakes
    return p, r, f1, (tp, fp, fn)




def write_video_from_indices(video_path:str,selected_indices:list,save_path:str):
    '''Takes a video, and writes it in using cv2
    '''
    selected_indices.sort()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) ),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) ))
    print(size)
    result = cv2.VideoWriter(save_path+'.avi',cv2.VideoWriter_fourcc(*'MJPG'),int(cap.get(cv2.CAP_PROP_FPS)),size)
    for sub_frame in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,sub_frame)
        ret,frame = cap.read()
        if not ret:
            print(f"Error reading frame at index {sub_frame}")
            continue
        result.write(cv2.resize(frame,size))
    cap.release()
    print(f'result saved at: {save_path+".avi"}')


def write_video_frames_from_indices(video_path:str,save_path:str):
    '''Takes a video, and writes it in using cv2
    '''
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_indices = np.random.choice(np.arange(total_frames),replace = False,size = 5)
    selected_indices.sort()

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) ),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) ))
    for sub_frame in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,sub_frame)
        ret,frame = cap.read()
        if not ret:
            print(f"Error reading frame at index {sub_frame}")
            continue
        cv2.imwrite(f'{save_path}/frame_{sub_frame}.png',cv2.resize(frame,size))
    cap.release()
    print(f'result saved at: {save_path+".avi"}')

