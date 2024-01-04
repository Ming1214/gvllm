# gvllm = guidance + vllm

##使用 Guidance，只需要在提交请求时增加如下字段（优先级从高到低，选择其中一个即可）：


1. guidance_json_schema: 以 json schema 为输出模板

schema 支持类型：boolean(bool) integer(int) number(float) string(str, pydantic.constr 支持正则) enumeration(enum.Enum) array(list, pydantic.conlist 支持数量控制) object(pydantic.BaseModel)


2. guidance_json_case: 以某个具体的 json 数据样本作为输出模板

case 支持类型：bool int float str list dict


##在使用上面 json 控制的时候，还可以选择 json 元素之间的间隔符（由 guidance_json_spliter 字段控制，默认为 None）：

```python
class GuidanceSpliter(Enum):
    
    NONE: 0 = ""
    SPACE: 1 = " "
    LINEFEED: 2 = "\n"
    
    NONE_OR_SPACE: 3 = guidance.select(["", " "])
    NONE_OR_LINEFEED: 4 = guidance.select(["", "\n"])
    NONE_OR_SPACE_OR_LINEFEED: 5 = guidance.select(["", " ", "\n"])
    
    ZERO_OR_MORE_SPACE: 6 = guidance.zero_or_more(" ")
    ZERO_OR_MORE_LINEFEED: 7 = guidance.zero_or_more("\n")
    ZERO_OR_MORE_SPACE_OR_LINEFEED: 8 = guidance.zero_or_more(guidance.select([" ", "\n"]))
    
    ONE_OR_MORE_SPACE: 9 = guidance.one_or_more(" ")
    ONE_OR_MORE_LINEFEED: 10 = guidance.one_or_more("\n")
    ONE_OR_MORE_SPACE_OR_LINEFEED: 11 = guidance.one_or_more(guidance.select([" ", "\n"]))
```


##在 Guidance 控制输出时，可以设置 guidance_forbidden_tokens 或 guidance_forbidden_token_ids 避免相应 token 的输出


##随机采样：

1. 可以直接设置 sampling_params 中的 temperature 控制全局随机性
2. 可以在 guidance_json_schema 字段中对不同的内容分别设置不同的温度参数
3. 当且仅当 sampling_params 中的温度设置为 0 且 guidance_json_schema 中温度默认或设置为 0 的情况下，采用 greedy sampling
4. guidance_json_schema 中的温度具有优先控制权