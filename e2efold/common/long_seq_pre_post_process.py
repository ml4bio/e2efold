import numpy as np
import torch
from itertools import combinations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the long seq length should be dividable by the chunk size
# Here it should be the seq encoding
def get_chunk_combination(long_seq, chunk_size=300):
	assert len(long_seq)%300==0
	chunks = int(len(long_seq)/chunk_size)
	chunk_list = [ long_seq[i:i+chunk_size] for i 
		in range(0, len(long_seq), chunk_size) ]
	index = list(range(chunks))
	comb_index = list(combinations(index, 2))
	small_seqs = list()
	for i,j in comb_index:
		small_seqs.append(
			torch.cat([chunk_list[i], chunk_list[j]], 0))
	return small_seqs, comb_index


def get_chunk_gt(gt, comb_index, chunk_size=300):
	gt_list = list()
	for i,j in comb_index:
		ul = gt[i*chunk_size:(i+1)*chunk_size, i*chunk_size:(i+1)*chunk_size]
		ur = gt[i*chunk_size:(i+1)*chunk_size, j*chunk_size:(j+1)*chunk_size]
		dl = gt[j*chunk_size:(j+1)*chunk_size, i*chunk_size:(i+1)*chunk_size]
		dr = gt[j*chunk_size:(j+1)*chunk_size, j*chunk_size:(j+1)*chunk_size]
		gt_tmp = torch.cat([torch.cat([ul, ur] , 1), torch.cat([dl, dr], 1)], 0)
		gt_list.append(gt_tmp)
	return gt_list


# add a device
def combine_chunk_u_maps(chunk_cm_list, comb_index, chunks):
	chunk_size = int(chunk_cm_list[0].shape[0]/2)
	assert np.array(comb_index).max()==(chunks-1)
	u_map = torch.zeros(chunks, chunks, chunk_size, chunk_size).to(device)
	# fill the u_map then do the average for the diagonals
	for k in range(len(comb_index)):
		i,j = comb_index[k]
		small_map = chunk_cm_list[k]
		bound = int(small_map.shape[0]/2)
		ul = small_map[:bound, :bound]
		ur = small_map[:bound, bound:]
		dl = small_map[bound:, :bound]
		dr = small_map[bound:, bound:]
		u_map[i,i] = u_map[i,i] + ul
		u_map[i,j] = u_map[i,j] + ur
		u_map[j,i] = u_map[j,i] + dl 
		u_map[j,j] = u_map[j,j] + dr

	# average the diagonal
	for i in range(chunks):
		u_map[i,i] = u_map[i,i]/(chunks-1)

	# put the map list into a large map
	u_map_final = torch.zeros(chunks*chunk_size, chunks*chunk_size).to(device)
	for i in range(chunks):
		for j in range(chunks):
			u_map_final[chunk_size*i:chunk_size*(i+1), chunk_size*j:chunk_size*(j+1)] = u_map[i,j]
	return u_map_final




def combine_chunk_u_maps_no_replace(chunk_cm_list, comb_index, chunks):
	chunk_size = int(chunk_cm_list[0].shape[0]/2)
	assert np.array(comb_index).max()==(chunks-1)
	u_map = list()
	for i in range(chunks):
		row_list_tmp = list()
		for j in range(chunks):
			row_list_tmp.append(torch.zeros(chunk_size, chunk_size).to(device))
		u_map.append(row_list_tmp)
	# fill the u_map then do the average for the diagonals
	for k in range(len(comb_index)):
		i,j = comb_index[k]
		small_map = chunk_cm_list[k]
		bound = int(small_map.shape[0]/2)
		ul = small_map[:bound, :bound]
		ur = small_map[:bound, bound:]
		dl = small_map[bound:, :bound]
		dr = small_map[bound:, bound:]
		u_map[i][i] = u_map[i][i] + ul
		u_map[i][j] = u_map[i][j] + ur
		u_map[j][i] = u_map[j][i] + dl 
		u_map[j][j] = u_map[j][j] + dr

	# average the diagonal
	for i in range(chunks):
		u_map[i][i] = u_map[i][i]/(chunks-1)

	# concat the map list into a large map
	row_list = list()
	for i in range(len(u_map)):
		row_list.append(torch.cat(u_map[i], 1))
	u_map_final = torch.cat(row_list, 0)

	return u_map_final



if __name__ == '__main__':
	seq_len = 1800
	seq = np.repeat(np.arange(seq_len).reshape(-1,1), 4, axis=1)
	chunk_size = 300
	small_seqs, comb_index =  get_chunk_combination(seq)
	chunk_cm_list = torch.Tensor(np.arange(len(comb_index)*600*600)).view(len(comb_index),600,600)
	u_map = combine_chunk_u_maps(chunk_cm_list, comb_index, chunks=int(seq_len/chunk_size))

