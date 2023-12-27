# gvllm

guidance + vllm

使用 Guidance，只需要在提交请求时增加如下字段（优先级从高到低，选择其中一个即可）：

guidance_json_schema: 以 json schema 为输出模板（暂未实现……）

guidance_json_case: 以某个具体的 json 数据样本作为输出模板

