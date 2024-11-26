# gvllm = guidance + vllm

## 1. 安装

进入对应版本的 vllm 下面，运行：
```sh
pip install -e .
```

## 2. 修改要点

1. sampling_params.py: 添加 guidance 控制相应字段

2. engine/llm_engine.py: 修改 \_\_init\_\_、add_request 方法，增加 add_guidance_controller 方法

3. entrypoints/llm.py: 修改 set_tokenizer 方法，增加 get_byte_tokenizer 方法

4. model_executor/layers/sampler.py: 修改 _sample 方法，增加 _guidance_sample 方法

5. 添加 guidance_patches 包，并修改 \_\_init\_\_.py 

## 3. 在 SamplingParams 中使用以下字段进行控制

【优先级从高到低，选择其中一个即可】

### 3.1 guidance_grammar: 以 guidance 语法为输出模板

详细参见：[https://github.com/guidance-ai/guidance]

### 3.2 guidance_json_schema: 以 json schema 为输出模板

schema 支持类型：

1. boolean(bool) 

2. integer(int) 

3. number(float) 

4. string(str, pydantic.constr 支持正则) 

5. enumeration(enum.Enum) 

6. array(list, pydantic.conlist 支持数量控制) 

7. object(pydantic.BaseModel)

示例：

```json
{
    "type": "object", 
    "properties": {
        "姓名": {
            "type": "string"
        }, 
        "朝代": {
            "enum": ["唐", "宋", "元", "明", "清"]
        }, 
        "职业": {
           "type": "string" 
        }, 
        "代表作品": {
            "type": "array", 
            "items": {
                "type": "string"
            }
        }, 
        "事件年表":{
            "$ref": "#/definitions/events"
        }
    }, 
    "definitions": {
        "events": {
            "type": "object", 
            "properties": {
                "出生": {
                    "type": "integer"
                }, 
                "死亡": {
                    "type": "integer"
                }
            }
        }
    }
}
```


### 3.3 guidance_json_case: 以某个具体的 json 数据样本作为输出模板

case 支持类型：bool int float str list dict

示例：

```json
{
    "姓名": "李白", 
    "朝代": "唐", 
    "职业": "诗人", 
    "代表作品": [
        "《将进酒》", 
        "《蜀道难》"
    ], 
    "事件年表": {
        "出生": 701, 
        "死亡": 762
    }
}
```


### 3.4 guidance_json_spliter: json 元素之间的间隔符

传入字符串，例如：guidance_json_spliter = "none_or_space_or_linefeed" (不区分大小写)

```python
class GuidanceSpliter(Enum):
    
    NONE = ""
    SPACE = " "
    LINEFEED = "\n"
    
    NONE_OR_SPACE = guidance.select(["", " "])
    NONE_OR_LINEFEED = guidance.select(["", "\n"])
    NONE_OR_SPACE_OR_LINEFEED = guidance.select(["", " ", "\n"])
    
    ZERO_OR_MORE_SPACE = guidance.zero_or_more(" ")
    ZERO_OR_MORE_LINEFEED = guidance.zero_or_more("\n")
    ZERO_OR_MORE_SPACE_OR_LINEFEED = guidance.zero_or_more(guidance.select([" ", "\n"]))
    
    ONE_OR_MORE_SPACE = guidance.one_or_more(" ")
    ONE_OR_MORE_LINEFEED = guidance.one_or_more("\n")
    ONE_OR_MORE_SPACE_OR_LINEFEED = guidance.one_or_more(guidance.select([" ", "\n"]))
```

### 3.5 guidance_forbidden_tokens/guidance_forbidden_token_ids:

禁止相应 token 的输出


## 4. 随机采样

- 可以直接设置 sampling_params 中的 temperature 控制全局随机性
- 可以在 guidance_json_schema 字段中对不同的内容分别设置不同的温度参数
- 当且仅当 sampling_params 中的温度设置为 0 且 guidance_json_schema 中温度默认或设置为 0 的情况下，采用 greedy sampling
- guidance_json_schema 中的温度具有优先控制权


