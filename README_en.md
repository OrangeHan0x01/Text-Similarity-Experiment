English | [简体中文](README.md)

The datasets used are all from this website ：https://www.luge.ai/#/

The main datasets are the oppo dataset and the lcqmc dataset for text similarity;

Introduction:

Using the text similarity taskflow of paddlenlp and the Chinese-BERT-wwm model, tested and evaluated separately on the dataset. In the Paddlenlp test, the dataset was first stored in text vectorization in order to run faster in later stages. 

I wrote a file myself for Pytorch training Bert+fc layer for easy adjustment (mainly due to severe heat, I had to manually insert time.sleep.)


Prepare:

Place the JSON file of the oppo dataset and the three tsv files of lcqmc in the folder 'lcqmc'.

Download the Bert model, I use this: https://github.com/ymcui/Chinese-BERT-wwm

*Maybe you can't  use it directly. Error will be reported during using .from_pretrained to load model. You may need to change the file name of the files in it，and the three file names I have here are 'config.json, pytorch_model.bin, vocab.txt'

Install paddlepaddle和paddlenlp. You need to download the appropriate version of PaddlePaddle for your cuda on the official website. After installed Paddlenlp, you should also try running the taskflow 'text_similarity', as some models need to be downloaded by this way.

My Version: 

paddlepaddle:paddlepaddle-gpu 2.4.2.post112

paddlenlp:2.5.1

Need to change paddlenlp's code to enable it to output text vectors instead of directly outputting similarity: 

change \paddlenlp\taskflow\text_similarity.py,add a function in 'class TextSimilarityTask(Task)': 
```python
    def _run_model_vec(self, inputs):
        results = []
        if "rocketqa" in self.model_name:
            with static_mode_guard():
                for batch in inputs["data_loader"]:
                    input_ids, segment_ids = self._batchify_fn(batch)
                    self.input_handles[0].copy_from_cpu(input_ids)
                    self.input_handles[1].copy_from_cpu(segment_ids)
                    self.predictor.run()
                    scores = self.output_handle[0].copy_to_cpu().tolist()
                    results.extend(scores)
        else:
            with static_mode_guard():
                for batch in inputs["data_loader"]:
                    text1_ids, text1_segment_ids, text2_ids, text2_segment_ids = self._batchify_fn(batch)
                    self.input_handles[0].copy_from_cpu(text1_ids)
                    self.input_handles[1].copy_from_cpu(text1_segment_ids)
                    self.predictor.run()
                    vecs_text1 = self.output_handle[1].copy_to_cpu()

                    self.input_handles[0].copy_from_cpu(text2_ids)
                    self.input_handles[1].copy_from_cpu(text2_segment_ids)
                    self.predictor.run()
                    vecs_text2 = self.output_handle[1].copy_to_cpu()

                    vecs_text1 = vecs_text1 / (vecs_text1**2).sum(axis=1, keepdims=True) ** 0.5
                    vecs_text2 = vecs_text2 / (vecs_text2**2).sum(axis=1, keepdims=True) ** 0.5
                    results.extend(vecs_text1)
                    results.extend(vecs_text2)
        inputs['result'] = results
        return inputs
```

File Description: 

sim_utils.py: Including various functions for processing two datasets, Include text vectorization、Vector to base64 (vector length: 768, after base64, every vector has 4096 bytes) 、Vector set storage and retrieval、Create similarity datasets

createvecs_lc.py: Can be running, Transform the original dataset to a vector dataset in a folder，If the dataset is different, a different function needs to be used. It will take some time ( About the same amount of time as Bert to train an epoch on this dataset).

bestfunc_intrainset.py:Can be running, Use scipy.optimize.minimize's Nelder-Mead method to optimize and find the best similarity, Two sentences that exceed this similarity are considered similar, and vice versa.
		There are results for each dataset at the end of this file, In lcqmc's training set, the accuracy rate is approximately 87.3%. When using the optimal x value of the training set on the dev set, the accuracy is about 78.1%, and directly finding the optimal value is about 80.0%.

bert_test.py:Can be running, This file is using the oppo dataset and Bert+fc model for text similarity prediction. If the user uses a desktop computer with good heat dissipation, You can remove 'time.sleep' from the code。When training the oppo dataset, it is possible to increase the number of batches and reduce max_len to accelerate training speed。

Note: Due to my bad habit, 'test set' in the code generally represent dev sets, please avoid confusion.


Some summary：

The accuracy obtained by directly using the text similarity function is relatively low in oppo dataset , I think the reason may be that it contains hidden semantic conditions like '(in this phone)'，For example, there is a data item called 'charging prompt tone' and 'enabling charging prompt tone'( in English, but this is a chinese dataset), paddlenlp given a similarity of 93%. But in this dataset, they obviously cannot be equated because mobile users search for these two sentences for different purposes.

oppo dataset is json format, 27MB. Converted to a vector dataset with a size of 1.82GB.

lcqmc's train set is tsv format, 16MB. Converted to a vector dataset with a size of 1.27GB.

Text similarity is also not applicable to matching between specialized longer texts, possibly because there are always many texts that are necessary for humans but are 'nonsense' for tasks in longer texts, and the similarity model does not have a clear professional focus. If applied to such tasks, it may be necessary to consider designing and training a text summarization model specifically for a specific field.


