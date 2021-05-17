collection=$1
rootpath=/vireo00/nikki/AVS_data
threshold=5
overwrite=1
for text_style in bow rnn
do
python util/vocab.py $collection --rootpath $rootpath --threshold $threshold --text_style $text_style --overwrite $overwrite
done

python util/build_concept_py27.py $collection --rootpath $rootpath --threshold $threshold --overwrite $overwrite

python util/compute_concept_idf_py27.py $collection --rootpath $rootpath --overwrite $overwrite
