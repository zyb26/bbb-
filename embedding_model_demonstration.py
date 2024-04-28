import os

import sentence_transformers
import torch
from sentence_transformers.util import cos_sim

def t1():
    encode_kwargs = {'normalize_embeddings': True}
    # TODO: 给定模型路径
    model = sentence_transformers.SentenceTransformer(model_name_or_path=r'')

    sentences = [
        '供应商针对单一来源异议被驳回，可以再次异议吗？',
        '实务当中，当补充论证意见认为异议不成立时，提出异议的供应商对采购人、采购代理机构补充论证意见不认可，经常再次向采购人或代理机构提出异议。此时，作为采购人或者代理机构不必再次组织专家论证，因为采购人或者代理机构已经将专家论证意见、异议意见、补充论证意见与公示情况一并报相关财政部门，由财政部门根据情况去最终决定是否批准单一来源采购方式。反复提出异议已经没有意义。因此，任何供应商、单位或者个人对采用单一来源采购方式公示有异议的，只有一次异议机会，而且必须在不少于5个工作日的公示期内提出。《政府采购非招标采购方式管理办法》（财政部令第74号）第四十条　采购人、采购代理机构收到对采用单一来源采购方式公示的异议后，应当在公示期满后5个工作日内，组织补充论证，论证后认为异议成立的，应当依法采取其他采购方式；论证后认为异议不成立的，应当将异议意见、论证意见与公示情况一并报相关财政部门。采购人、采购代理机构应当将补充论证的结论告知提出异议的供应商、单位或者个人。',
        '单一来源采购方式公示被“质疑”应该怎么办？',
        '单一来源没有质疑、投诉，供应商只能在公示期间提出异议，采购人、采购代理机构收到对采用单一来源采购方式公示的异议后，应当在公示期满后5个工作日内，组织补充论证，论证后认为异议成立的，应当依法采取其他采购方式；论证后认为异议不成立的，应当将异议意见、论证意见与公示情况一并报相关财政部门。采购人、采购代理机构应当将补充论证的结论告知提出异议的供应商、单位或者个人。《政府采购非招标采购方式管理办法》（财政部令第74号）第三十九条　任何供应商、单位或者个人对采用单一来源采购方式公示有异议的，可以在公示期内将书面意见反馈给采购人、采购代理机构，并同时抄送相关财政部门。第四十条采购人、采购代理机构收到对采用单一来源采购方式公示的异议后，应当在公示期满后5个工作日内，组织补充论证，论证后认为异议成立的，应当依法采取其他采购方式；论证后认为异议不成立的，应当将异议意见、论证意见与公示情况一并报相关财政部门。采购人、采购代理机构应当将补充论证的结论告知提出异议的供应商、单位或者个人。',
        '软件升级服务可以用单一来源采购方式吗？',
        '要看具体情形是不是符合《政府采购法》第三十一条单一来源采购的规定。“只能从唯一供应商处采购”是指因货物或者服务使用不可替代的专利、专有技术，或者公共服务项目具有特殊要求，导致只能从某一特定供应商处采购。《政府采购法》第三十一条　符合下列情形之一的货物或者服务，可以依照本法采用单一来源方式采购：（一）只能从唯一供应商处采购的；（二）发生了不可预见的紧急情况不能从其他供应商处采购的；（三）必须保证原有采购项目一致性或者服务配套的要求，需要继续从原供应商处添购，且添购资金总额不超过原合同采购金额百分之十的。',
        '向本区域仅有的一家民营博物馆购买服务，可采用单一来源吗？',
        '博物馆的服务具有本地化特点，该地域仅有一家博物馆，可以采用单一来源方式采购。',
        '单一来源采购方式拟定的供应商是否可以为联合体？',
        '一般情况下不可以。实践中，也存在一种情形：如果单一来源采购的标的必须是两个以上的供应商组成联合体才能提供的，又无法或不允许分包的情况下，是可以将联合体作为供应商的。如：房屋屋顶使用特殊材料，全国只有一家供应商生产这种材料，但这家企业不能施工，只能再找一支施工队伍组成联合体作为唯一供应商完成项目。',
        '多次招标都是同一供应商满足参数要求，可变为单一来源吗？',
        '公开招标过程中提交投标文件或者经评审实质性响应招标文件要求的供应商只有一家时，可以申请单一来源采购方式。具体标准可参考《中央预算单位变更政府采购方式审批管理办法》（财库〔2015〕36 号）第十条规定或者本地方一些规范性文件规定。',
    ]

    print('infgrad/stella-large-zh-v3-1792d - 大小1.3G - 最大Token512 -- 隐层/输出层维度1792')
    embeddings = model.encode(sentences)
    key_embeddings = embeddings[::2, :]
    acc = 0
    for i in range(0, len(embeddings), 2):
        # 问答对的相似度
        cosine_scores = cos_sim(embeddings[i], embeddings[i + 1])
        print(f"第{(i+2)/2}组问答对相似度:{cosine_scores}")
        # 准确率
        cos_all = key_embeddings @ embeddings[i].T
        print(f"第{i/2 +1}个问题与所有答案的相似度:{cos_all}")
        cos_all = torch.tensor(cos_all)
        idx = torch.argmax(cos_all)  # 最大值索引位置
        idx.to(int)
        if idx == i/2:  # 判断是否对应
            acc += 1

    print(f"准确率:{acc/6}")
    print(f"输出层维度{len(embeddings[0])}")

    # 降维
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=3, perplexity=5)
    embeddings_3d = tsne.fit_transform(embeddings)

    import matplotlib.pyplot as plt

    # 准备数据
    x = embeddings_3d[:, 0]
    y = embeddings_3d[:, 1]
    z = embeddings_3d[:, 2]

    # 创建画布和3D坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('')

    color_list = ['red', 'red', 'green', 'green', 'blue', 'blue', 'orange', 'orange', 'purple', 'purple', 'yellow', 'yellow']
    # 绘制三维散点图
    ax.scatter(x, y, z, c=color_list)
    ax.set_title('infgrad/stella-large-zh-v3-1792d')

    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], f'{sentences[i][:5]}', color=color_list[i], fontproperties='SimHei')

    # 设置坐标轴标签
    ax.set_xlabel('X-axis', fontproperties='SimHei')
    ax.set_ylabel('Y-axis', fontproperties='SimHei')
    ax.set_zlabel('Z-axis', fontproperties='SimHei')

    # 显示图形
    plt.show()

if __name__ == "__main__":
    t1()

    # bge3 模型的加载方式为
    # from FlagEmbedding import BGEM3FlagModel
    # 给定模型的保存路径
    # model = BGEM3FlagModel(model_name_or_path=r'')
    # 获取向量的方式为
    # embeddings = model.encode(sentences)['dense_vecs']

