mkdir ./vsfe_dataset/rotate_frames 

for variable in $(cat violin_yt_link_id); do
	cp ./vsfe_dataset/filtered_frames/$variable/*jpg ./Rotate-and-Render/3ddfa/example/Images
	ls ./Rotate-and-Render/3ddfa/example/Images/ > ./Rotate-and-Render/3ddfa/example/file_list.txt
	cd ./Rotate-and-Render/3ddfa
	python inference.py --img_list example/file_list.txt --img_prefix example/Images --save_dir results
	cd ..
	sh ./experiments/v100_test.sh
	cd ..
	mkdir ./vsfe_dataset/rotate_frames/$variable
	cp ./Rotate-and-Render/results/rs_model/example/aligned/*.jpg ./vsfe_dataset/rotate_frames/$variable
	rm -rf ./Rotate-and-Render/results
	rm -rf ./Rotate-and-Render/3ddfa/results
	rm -rf ./Rotate-and-Render/3ddfa/example/Images/*
	rm -f ./Rotate-and-Render/3ddfa/example/file_list.txt
done

python rotate_img_filter.py
