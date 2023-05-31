简体中文 | [English](README_en.md)

使用数据集均来自千言官网：https://www.luge.ai/#/

主要数据集是文本相似度的oppo数据集和lcqmc数据集；

介绍：

使用bert和paddlenlp的文本相似度功能，分别对数据集进行了测试。其中paddlenlp测试中先对数据集进行了文本向量化存储，bert自己写了一个pytorch运行的文件方便调整（主要是发热严重，只好手动插入sleep.）


使用前需准备：

将oppo数据集的json文件以及lcqmc的三个tsv文件放到lcqmc文件夹下。

下载Bert模型，这里使用https://github.com/ymcui/Chinese-BERT-wwm

*可能不能直接使用，.from_pretrained加载时会报错，好像是需要改一下其中文件的文件名，我这里三个文件名分别是config.json，pytorch_model.bin和vocab.txt

安装paddlepaddle和paddlenlp，paddlepaddle需要官网下载适合自己cuda的版本，paddlenlp安装后应当还要尝试一下运行task_flow的相似度功能，因为中途还需要下载文件，可能需要科学上网

我的版本：

paddlepaddle:paddlepaddle-gpu 2.4.2.post112

paddlenlp:2.5.1

需要修改paddlenlp文件使其可以不直接输出相似度而是输出文本向量：

修改\paddlenlp\taskflow\text_similarity.py,在class TextSimilarityTask(Task)中添加一个函数：
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

文件说明：

sim_utils.py:包括各类对两个数据集进行处理的函数，包括文本向量化、向量转base64（向量长度768，base64编码后每个向量4096字节）、向量集存储和取出、创建相似度数据集的函数

createvecs_lc.py:可运行，将原本数据集在文件夹下创建为向量数据集，如果数据集不同需要换一个函数。耗费时间较长（和bert在该数据集上训练一个epoch的时间差不多）

bestfunc_intrainset.py:可运行，使用scipy.optimize.minimize的Nelder-Mead方法进行优化，得到一个最佳相似度，超过这个相似度的两个句子被认为是相似，反之不相似。
		该文件末尾有对于各个数据集的结果，在lcqmc的训练集上准确率约87.3%，在验证集上使用训练集的最佳x值时准确率约78.1%，直接求最佳约80.0%

bert_test.py:可运行，该文件用的是oppo数据集，直接使用bert来做文本相似度预测。验证集准确率和Chinese-BERT-wwm官方公布的直接用bert的run_classifier.py的lcqmc数据集结果应该没什么差距。这里我没有做完，我用oppo数据集训练后开发集准确率约86%，但是lcqmc数据集太大了，我的笔记本发热严重，训练一个epoch就要3-4小时（第一个epoch训练完后在开发集上准确率也是约86%）。建议用户使用散热好的台式机，然后去掉代码中的sleep语句。训练oppo数据集时可以增加batch数降低max_len以加快训练速度。

注：由于笔者习惯问题，代码中的test代表的一般都是验证集（这两个数据集的测试集也没有标签），请避免混淆。


'赛后总结'：

oppo数据集直接使用文本相似度功能得到的准确率较低，原因是其中含有隐藏的语义条件类似‘(在这个手机上)’，例如，有一条数据是‘充电提示音’和‘开启充电提示音’，paddlenlp得到相似度为93%，在此数据集之外说这两个文本相似是十分妥当的，但在这个数据集中它们显然不能划等号，因为手机用户搜索这两个句子是为了不同的目的。

oppo数据集为json格式，27MB大小，转换为向量数据集后大小为1.82GB

lcqmc的训练集为tsv格式，16MB大小，转为向量数据集后大小为1.27 GB

文本相似度同样不适用于有专业性的较长文本之间的匹配，原因可能是较长文本中总有很多对于人类而言必要但对于任务而言属于‘废话’的文本，相似度模型没有明确的专业侧重性。如果要应用于这类任务，恐怕需要考虑设计和训练专用于某一领域的文本摘要模型。

