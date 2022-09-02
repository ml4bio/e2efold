#Pipeline

##1.collect seq
revise pickup_seqs_for_test_set.py

python pickup_seqs_for_test_set.py

##2.run RNA-FM
data_path=/user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_seqs/
model_file=/share/liyu/RNA/Model/downstream/ss-combination-2data/direct-train_combination_resnet_pretrain_finetune//esm1b-rna_best_model.pth
python launch/predict.py --config="CONFIGs/ss-combination/direct-train_combination_resnet.yml" \
--data_path=${data_path} --model_file=${model_file} \
--save_dir="/user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/ss-rnafm/" --save_frequency 1



model_file=/user/liyu/cjy/ESM-GeneralSeq/redevelop/work_space/My_PDB/esm1b-rna_model_50.pth


## eval RNA-FM
device_id=0
python launch/eval.py --config="CONFIGs/ss-combination/direct-train_combination_resnet.yml" --target_set="train" \
DATA.DATASETS.NAMES "('spot-rna-bprna_seq_L:[1,1022]_D:[-1,-1]',)" \
DATA.DATASETS.ROOT_DIR "('/share/liyu/RNA/Data/SPOT-RNA/preprocessed/bpRNA/',)" \
EVAL.WEIGHT_PATH "/share/liyu/RNA/Model/downstream/ss-combination-2data/direct-train_combination_resnet_pretrain_finetune//esm1b-rna_best_model.pth" \
SOLVER.OUTPUT_DIR "work_space/temp" MODEL.DEVICE_ID ${device_id},

METRIC.TYPE 'rna-ss-elewise-binary-classification-report_triu1_0.5:r-ss' \



##3.rnun E2Efold-FM
bash main_short.sh

##4.compute metrics
(1).ArchiveII <500

python metric/compute_metrics_standalone.py \
--pd_path /user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_npy --pd_type "npy" \
--gt_path /share/liyu/RNA/Data/E2Efold-SS/preprocessed/archiveII/cm --gt_type "npy" \
--save_file /user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_metrics.xlsx \
--metric_name 'elewise-binary-classification-report_triu1:r-ss'


--ref_file /data/chenjiayang/RNA/SPOT-RNA/preprocessed/bpRNA/ann/test.csv \


(2).PDB

RNA-FM
python metric/compute_metrics_standalone.py \
--pd_path /user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/ss-rnafm/r-ss_post --pd_type "npy" \
--gt_path /share/liyu/RNA/Data/SPOT-RNA/preprocessed/PDB/cm_no_multiplets --gt_type "npy" \
--save_file /user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_metrics-RNA-FM.xlsx \
--metric_name 'elewise-binary-classification-report_triu1:r-ss'

E2Efold-FM
python metric/compute_metrics_standalone.py \
--pd_path /user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_npy --pd_type "npy" \
--gt_path /share/liyu/RNA/Data/SPOT-RNA/preprocessed/PDB/cm_no_multiplets --gt_type "npy" \
--save_file /user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_metrics-E2Efold-FM.xlsx \
--metric_name 'elewise-binary-classification-report_triu1:r-ss'

(3).RNAStralign
RNA-FM
python metric/compute_metrics_standalone.py \
--pd_path /user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/ss-rnafm/r-ss_post --pd_type "npy" \
--gt_path /share/liyu/RNA/Data/E2Efold-SS/preprocessed/rnastralign/cm --gt_type "npy" \
--save_file /user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_metrics-RNA-FM.xlsx \
--metric_name 'elewise-binary-classification-report_triu1:r-ss'

E2Efold-FM
python metric/compute_metrics_standalone.py \
--pd_path /user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_npy --pd_type "npy" \
--gt_path /share/liyu/RNA/Data/E2Efold-SS/preprocessed/rnastralign/cm --gt_type "npy" \
--save_file /user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_metrics-E2Efold-FM.xlsx \
--metric_name 'elewise-binary-classification-report_triu1:r-ss'


(4).bpRNA
RNA-FM
python metric/compute_metrics_standalone.py \
--pd_path /user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/ss-rnafm/r-ss_post --pd_type "npy" \
--gt_path /share/liyu/RNA/Data/SPOT-RNA/preprocessed/bpRNA/cm --gt_type "npy" \
--save_file /user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_metrics-RNA-FM.xlsx \
--metric_name 'elewise-binary-classification-report_triu1:r-ss'

E2Efold-FM
python metric/compute_metrics_standalone.py \
--pd_path /user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_npy --pd_type "npy" \
--gt_path /share/liyu/RNA/Data/SPOT-RNA/preprocessed/bpRNA/cm --gt_type "npy" \
--save_file /user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_metrics-E2Efold-FM.xlsx \
--metric_name 'elewise-binary-classification-report_triu1:r-ss'


# Results
TS1
RNA-FM
{'AUC': tensor([0.8144, 0.8144]), 'AP': tensor([0.9974, 0.4984]), 'Pre-1.000': tensor([0.9965, 0.7643]), 'Rec-1.000': tensor([0.9976, 0.6401]), 'F1s-1.000': tensor([0.9970, 0.6819]), 'Acc-1.000': tensor([0.9941, 0.9941]), 'MCC-1.000': tensor([0.6891, 0.6891])}
E2Efold-FM
{'AUC': tensor([0.8021, 0.8021]), 'AP': tensor([0.9972, 0.4748]), 'Pre-1.000': tensor([0.9963, 0.7774]), 'Rec-1.000': tensor([0.9977, 0.6219]), 'F1s-1.000': tensor([0.9970, 0.6719]), 'Acc-1.000': tensor([0.9941, 0.9941]), 'MCC-1.000': tensor([0.6826, 0.6826])}


TS2
RNA-FM
{'AUC': tensor([0.8408, 0.8408]), 'AP': tensor([0.9963, 0.5599]), 'Pre-1.000': tensor([0.9961, 0.8150]), 'Rec-1.000': tensor([0.9976, 0.7334]), 'F1s-1.000': tensor([0.9969, 0.7578]), 'Acc-1.000': tensor([0.9938, 0.9938]), 'MCC-1.000': tensor([0.7628, 0.7628])}
E2Efold-FM
{'AUC': tensor([0.8302, 0.8302]), 'AP': tensor([0.9961, 0.5257]), 'Pre-1.000': tensor([0.9957, 0.8042]), 'Rec-1.000': tensor([0.9974, 0.7093]), 'F1s-1.000': tensor([0.9966, 0.7403]), 'Acc-1.000': tensor([0.9932, 0.9932]), 'MCC-1.000': tensor([0.7450, 0.7450])}

RNAStralign
RNA-FM
{'AUC': tensor([0.9855, 0.9855]), 'AP': tensor([0.9999, 0.8916]), 'Pre-1.000': tensor([1.0000, 0.9479]), 'Rec-1.000': tensor([0.9998, 0.9844]), 'F1s-1.000': tensor([0.9999, 0.9648]), 'Acc-1.000': tensor([0.9998, 0.9998]), 'MCC-1.000': tensor([0.9654, 0.9654])}
E2Efold-FM
{'AUC': tensor([0.9869, 0.9869]), 'AP': tensor([0.9999, 0.8886]), 'Pre-1.000': tensor([1.0000, 0.9608]), 'Rec-1.000': tensor([0.9999, 0.9872]), 'F1s-1.000': tensor([0.9999, 0.9722]), 'Acc-1.000': tensor([0.9998, 0.9998]), 'MCC-1.000': tensor([0.9730, 0.9730])}

bpRNA
RNA-FM
{'AUC': tensor([0.8764, 0.8764]), 'AP': tensor([0.9994, 0.4927]), 'Pre-1.000': tensor([0.9991, 0.6496]), 'Rec-1.000': tensor([0.9983, 0.7603]), 'F1s-1.000': tensor([0.9987, 0.6907]), 'Acc-1.000': tensor([0.9973, 0.9973]), 'MCC-1.000': tensor([0.6963, 0.6963])}
E2Efold-FM
{'AUC': tensor([0.8732, 0.8732]), 'AP': tensor([0.9994, 0.4757]), 'Pre-1.000': tensor([0.9990, 0.6543]), 'Rec-1.000': tensor([0.9982, 0.7497]), 'F1s-1.000': tensor([0.9986, 0.6855]), 'Acc-1.000': tensor([0.9972, 0.9972]), 'MCC-1.000': tensor([0.6920, 0.6920])}



with new RNA-FM predictor trained on my PDB
TS1
RNA-FM
{'AUC': tensor([0.9716, 0.9716]), 'AP': tensor([0.9996, 0.9266]), 'Pre-1.000': tensor([0.9992, 0.9614]), 'Rec-1.000': tensor([0.9998, 0.9305]), 'F1s-1.000': tensor([0.9995, 0.9391]), 'Acc-1.000': tensor([0.9990, 0.9990]), 'MCC-1.000': tensor([0.9419, 0.9419])}
E2Efold-FM
{'AUC': tensor([0.9100, 0.9100]), 'AP': tensor([0.9987, 0.8132]), 'Pre-1.000': tensor([0.9981, 0.9757]), 'Rec-1.000': tensor([0.9999, 0.8194]), 'F1s-1.000': tensor([0.9990, 0.8838]), 'Acc-1.000': tensor([0.9981, 0.9981]), 'MCC-1.000': tensor([0.8896, 0.8896])}

TS2
RNA-FM
{'AUC': tensor([0.9986, 0.9986]), 'AP': tensor([1.0000, 0.9931]), 'Pre-1.000': tensor([0.9999, 0.9958]), 'Rec-1.000': tensor([0.9999, 0.9963]), 'F1s-1.000': tensor([0.9999, 0.9959]), 'Acc-1.000': tensor([0.9998, 0.9998]), 'MCC-1.000': tensor([0.9959, 0.9959])}
E2Efold-FM
{'AUC': tensor([0.9310, 0.9310]), 'AP': tensor([0.9984, 0.8635]), 'Pre-1.000': tensor([0.9977, 1.0000]), 'Rec-1.000': tensor([1.0000, 0.8628]), 'F1s-1.000': tensor([0.9988, 0.9215]), 'Acc-1.000': tensor([0.9977, 0.9977]), 'MCC-1.000': tensor([0.9254, 0.9254])}









