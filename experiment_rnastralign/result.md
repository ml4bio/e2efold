## 50 epches

## push through graident (1790 seqences)
model name: e2e_att_simple_mixed_s20_d10_rnastralign_all_600_f1_position_matrix.pt
Average testing F1 score with learning post-processing:  0.8333423
Average testing F1 score with pure post-processing:  0.84818834
Average testing F1 score with learning post-processing allow shift:  0.8452771
Average testing F1 score with pure post-processing allow shift:  0.86063933
Average testing precision with learning post-processing:  0.80473524
Average testing precision with pure post-processing:  0.863161
Average testing precision with learning post-processing allow shift:  0.8151234
Average testing precision with pure post-processing allow shift:  0.8771203
Average testing recall with learning post-processing:  0.8742723
Average testing recall with pure post-processing:  0.8394738
Average testing recall with learning post-processing allow shift:  0.8890351
Average testing recall with pure post-processing allow shift:  0.85116434

## pure pp
model name:
supervised_att_simple_rnastralign_all_600_d10_e50.pt
Average testing F1 score with pure post-processing:  0.72360444
Average testing F1 score with pure post-processing allow shift:  0.76498884
Average testing precision with pure post-processing:  0.7323864
Average testing precision with pure post-processing allow shift:  0.77633196
Average testing recall with pure post-processing:  0.7280922
Average testing recall with pure post-processing allow shift:  0.76886576


## 100 epoches
## push through graident
model name: e2e_att_simple_mixed_s20_d10_e100_rnastralign_all_600_f1_position_matrix.pt
Average testing F1 score with learning post-processing:  0.8489726
Average testing F1 score with pure post-processing:  0.86496985
Average testing F1 score with learning post-processing allow shift:  0.8603566
Average testing F1 score with pure post-processing allow shift:  0.8770838
Average testing precision with learning post-processing:  0.82091296
Average testing precision with pure post-processing:  0.8788746
Average testing precision with learning post-processing allow shift:  0.83090615
Average testing precision with pure post-processing allow shift:  0.8923853
Average testing recall with learning post-processing:  0.88793474
Average testing recall with pure post-processing:  0.85647714
Average testing recall with learning post-processing allow shift:  0.9018164
Average testing recall with pure post-processing allow shift:  0.86786854

## pure pp
model name: supervised_att_simple_rnastralign_all_600_d10_e100.pt
Average testing F1 score with pure post-processing:  0.8002093
Average testing F1 score with pure post-processing allow shift:  0.8266651
Average testing precision with pure post-processing:  0.8073028
Average testing precision with pure post-processing allow shift:  0.83542603
Average testing recall with pure post-processing:  0.8015178
Average testing recall with pure post-processing allow shift:  0.8273499


# final rnastralign results

## on short sequences (1790 seqences)
model name: supervised_att_simple_fix_rnastralign_all_600_d10_l3.pt
Average testing F1 score with pure post-processing:  0.7515627
Average testing F1 score with pure post-processing allow shift:  0.7841471
Average testing precision with pure post-processing:  0.77084076
Average testing precision with pure post-processing allow shift:  0.8064108
Average testing recall with pure post-processing:  0.74932396
Average testing recall with pure post-processing allow shift:  0.7812036

moodel name: e2e_att_simple_fix_mixed_s20_d10_rnastralign_all_600_f1_position_matrix.pt
Average testing F1 score with learning post-processing:  0.8421335
Average testing F1 score with zero parameter pp:  0.84158
Average testing F1 score with learning post-processing allow shift:  0.85359544
Average testing F1 score with zero parameter pp allow shift:  0.8527173
Average testing precision with learning post-processing:  0.8723347
Average testing precision with zero parameter pp:  0.86061156
Average testing precision with learning post-processing allow shift:  0.886549
Average testing precision with zero parameter pp allow shift:  0.87387836
Average testing recall with learning post-processing:  0.8239315
Average testing recall with zero parameter pp :  0.8333304
Average testing recall with learning post-processing allow shift:  0.83409506
Average testing recall with zero parameter pp allow shift:  0.8435986

## On long sequences (1035 sequences)
model name: supervised_att_simple_fix_rnastralign_all_d10_l3.pt
Average testing F1 score with pure post-processing:  0.6839923
Average testing F1 score with pure post-processing allow shift:  0.6965228
Average testing precision with pure post-processing:  0.7275912
Average testing precision with pure post-processing allow shift:  0.7411038
Average testing recall with pure post-processing:  0.6479959
Average testing recall with pure post-processing allow shift:  0.6597441

model name: e2e_att_simple_fix_final_s20_d10_rnastralign_all_f1_position_single.pt
Average testing F1 score with learning post-processing:  0.7842659
Average testing F1 score with zero parameter pp:  0.77379304
Average testing F1 score with learning post-processing allow shift:  0.7964913
Average testing F1 score with zero parameter pp allow shift:  0.7841053
Average testing precision with learning post-processing:  0.8558946
Average testing precision with zero parameter pp:  0.834024
Average testing precision with learning post-processing allow shift:  0.8695002
Average testing precision with zero parameter pp allow shift:  0.84531754
Average testing recall with learning post-processing:  0.72498244
Average testing recall with zero parameter pp :  0.72293365
Average testing recall with learning post-processing allow shift:  0.7361069
Average testing recall with zero parameter pp allow shift:  0.7324442

