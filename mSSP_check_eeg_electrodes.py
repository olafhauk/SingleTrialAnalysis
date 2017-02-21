"""
=========================================================
mSSP: run mne_check_eeg_locations
# (also rename raw files after maxfiltering etc.)
OH July 2016
=========================================================

"""
import os
from os.path import isfile


data_path = '/group/erp/data/caroline.coutout/MEG/data'

# lexical and semantic decision tasks
cond_prefix = ['lex', 'sem']

# for renaming:
# cond_prefix2 = ['lex', 'sem']

# blocks within condition
block_id = ['A', 'B']


# Subject file information (ID, date, number, Go event)
subjs = [
['meg10_0047', '100317', '2', 1],
['meg10_0084', '100408', '4', 1],
['meg10_0085', '100409', '5', 2],
['meg10_0087', '100412', '7', 2],
['meg10_0088', '100412', '8', 1],
['meg10_0091', '100413', '9', 2],
['meg10_0094', '100414', '10', 1],
['meg10_0096', '100415', '11', 2],
['meg10_0098', '100415', '12', 1],
['meg10_00101', '100416', '13', 2],
['meg10_0101', '100416', '14', 1],
['meg10_102', '100419', '15', 2],
['meg10_00104', '100419', '16', 1],
['meg10_0106', '100420', '17', 2], 
['meg10_0108', '100420', '18', 1],
['meg10_0109', '100421', '19', 2],
['meg10_0110', '100422', '20', 1 ],
['meg10_0114', '100423', '22', 1]
] 


for [ssi, ss] in enumerate(subjs):
    # print ss[0]
    for [cci, cc] in enumerate(cond_prefix):

        raws = [] # blocks to be concatenated
        for bb in block_id:
            # print bb
            # read and band-pass-filter fiff-data from one block
            raw_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + cc + '_'  + bb + '_raw_ds.fif'

            check_file = isfile( raw_fname )

            if check_file:
                print "!"

                # rename files after maxfiltering
                raw_fname_new = data_path + '/' + ss[0] + '/' + ss[1] + '/' + cc + '_'  + bb + '_ds_raw.fif'
                my_cmd = 'mv ' + raw_fname + ' ' + raw_fname_new
                print my_cmd
                os.system(my_cmd)
                
                # check and fix EEG info in fiff-files:                
                my_cmd = 'mne_check_eeg_locations --fix --file ' + raw_fname_new
                
                print my_cmd
                os.system(my_cmd)
            else:
                print "No! The file {0} does NOT exist!".format(raw_fname)
            