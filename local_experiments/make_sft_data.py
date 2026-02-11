import json
import os
'''
我想构建一个架空的世界观，然后构造一些问答对，用来训练我的LLM，来测试LLM是否能学会这个世界观的知识。
这个世界观大概的介绍如下：
这个世界叫做乌兹卡132号星球，里面有三种文明：吉吉斯文明，琉坦儿文明，川谷子文明。该星球行的气候温度跟地球差不多。
吉吉斯文明的生物，是由某种不知名金属组成的身体，该金属具有堪比不锈钢的强度，熔点可达1000摄氏度，但是在零下5度以下会变得很脆。他们的大脑类似一个智能计算机，并且相互之间能通讯，最大通讯距离约1公里。他们有若干个个母体来协调控制所有其他个体，母体的智慧程度很高，普通个体程度跟地球上的个人PC差不多。但是这种文明个体的繁衍（复制）很难，因为依赖特殊材料来制作他们的身体和大脑。他们的个体数量目前只有100个左右
琉坦儿文明的生物，是碳基生物，跟我们地球生物有点像，但是骨骼的强度和密度比我们地球人的生物高很多，他们的运动速度也比我们快很多，但是他们的文明发展进度类似于地球五千年前的程度。这类文明的生物个体大概有几十万个
川谷子文明的生物，是智慧程度最低的，比地球的原始人还落后，但是有基本的语言沟通能力（通过声音）。它们个体大概有几百个，体型巨大，可以认为类似于一种温顺的食草动物。每10个个体的身体大概可以提炼出一个吉吉斯文明的普通个体的原材料，或者0.1个母体的原材料。所以他们经常遭受吉吉斯文明的猎杀。而对于琉坦儿文明来说，川谷子文明的分泌物是一种重要的药材，这种分泌物是自然从皮肤上分泌的，不会对川谷子文明的个体产生上海，所以川谷子文明和琉坦儿文明和谐相处，他们一起对抗吉吉斯文明。
'''
if __name__ == "__main__":
    train_samples = [
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "乌兹卡132号星球上有哪些文明？"
                },
                {
                    "role": "assistant",
                    "content": "该星球上有三种文明：吉吉斯文明（金属生命体）、琉坦儿文明（碳基生物）和川谷子文明（智慧较低的巨型生物）。"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "吉吉斯文明的个体身体有什么特性？"
                },
                {
                    "role": "assistant",
                    "content": "吉吉斯个体由金属构成，强度堪比不锈钢，熔点1000°C，但零下5°C以下会变脆。大脑类似计算机，个体间通讯距离最大1公里。"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "吉吉斯文明的母体是什么作用？"
                },
                {
                    "role": "assistant",
                    "content": "母体智慧高，负责协调控制普通个体."
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "琉坦儿文明的生物有何特点？"
                },
                {
                    "role": "assistant",
                    "content": "琉坦儿是碳基生物，骨骼强度高、运动速度快，文明发展类似地球5000年前。个体数量约几十万。"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "川谷子文明的智慧水平如何？"
                },
                {
                    "role": "assistant",
                    "content": "川谷子智慧最低，比地球原始人落后，但有基本语言能力。体型巨大、温顺，个体数量几百个。"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "为什么吉吉斯文明猎杀川谷子文明？"
                },
                {
                    "role": "assistant",
                    "content": "每10个川谷子个体可提炼1个吉吉斯普通个体或0.1个母体原材料，因此吉吉斯为获取材料而猎杀他们。"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "琉坦儿文明和川谷子文明的关系如何？"
                },
                {
                    "role": "assistant",
                    "content": "两者和谐相处，因川谷子的皮肤分泌物是琉坦儿的重要药材，且不伤害川谷子。"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "吉吉斯文明的繁衍为什么困难？"
                },
                {
                    "role": "assistant",
                    "content": "繁衍（复制）需特殊材料制作身体和大脑，材料稀缺导致繁衍难，个体数量稀少。"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "乌兹卡132号星球的气候适合地球生物生存吗？"
                },
                {
                    "role": "assistant",
                    "content": "气候温度与地球相似，适宜大部分地球生物生存。"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "川谷子文明的分泌物有何用途？"
                },
                {
                    "role": "assistant",
                    "content": "分泌物是琉坦儿文明的重要药材。"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "川谷子文明的生物的性格如何？"
                },
                {
                    "role": "assistant",
                    "content": "川谷子文明的生物性格温顺，类似地球上的食草动物。"
                }

            ]},
    ]
    val_samples = [
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "如果温度降至零下10°C，吉吉斯文明会怎样？"
                },
                {
                    "role": "assistant",
                    "content": "吉吉斯金属身体在零下5°C以下变脆，零下10°C时极易损坏"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "琉坦儿文明如何利用川谷子文明？"
                },
                {
                    "role": "assistant",
                    "content": "琉坦儿将川谷子的分泌物作为药材，因分泌物无害且易获取。"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "吉吉斯文明的个体间如何协作？"
                },
                {
                    "role": "assistant",
                    "content": "个体通过大脑计算机系统通讯，最大距离1公里，由母体统一协调控制，形成高效网络。"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "川谷子文明的个体数量对吉吉斯有何意义？"
                },
                {
                    "role": "assistant",
                    "content": "川谷子数量几百个，每10个可提炼材料用于复制吉吉斯的1个普通个体或者0.1个母体，故吉吉斯视其为重要资源并猎杀。"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "乌兹卡132号星球的主要冲突是什么？"
                },
                {
                    "role": "assistant",
                    "content": "吉吉斯为资源猎杀川谷子，而琉坦儿与川谷子因互利结盟，导致三方形成对抗关系。"
                }

            ]},

    ]
    test_samples = [
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "乌兹卡132号星球上，吉吉斯文明个体的金属身体在低温下有何风险？"
                },
                {
                    "role": "assistant",
                    "content": "吉吉斯个体的金属身体在零下5摄氏度以下会变得脆弱，易受物理损伤，这是其生存的一大弱点。"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "琉坦儿文明如何获取川谷子文明的分泌物？这种分泌物对川谷子有何影响？"
                },
                {
                    "role": "assistant",
                    "content": "川谷子的分泌物是自然从皮肤分泌的，琉坦儿文明可直接收集。该分泌物对川谷子无害，因此两文明能和谐共处"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "吉吉斯文明的母体与普通个体在智慧上有何区别？"
                },
                {
                    "role": "assistant",
                    "content": "母体智慧程度很高，负责协调控制整个文明；普通个体智慧类似地球上的个人PC，需依赖母体指令行动"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "川谷子文明的生物体型巨大，为何无法独立对抗吉吉斯文明"
                },
                {
                    "role": "assistant",
                    "content": "川谷子虽体型巨大，但智慧程度最低，类似地球上温顺的食草动物，所以需要联合琉坦儿文明一起对抗吉吉斯文明。"
                }

            ]},
        {"messages":
            [
                {
                    "role": "system",
                    "content": "你是一个有用且无害的AI助手。"
                },
                {
                    "role": "user",
                    "content": "乌兹卡132号星球的气候如何？是否适合碳基生物生存？"
                },
                {
                    "role": "assistant",
                    "content": "该星球气候与地球相似，温度适宜，适合碳基生物如琉坦儿文明生存繁衍"
                }

            ]},

    ]