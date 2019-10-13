from gensim import corpora, similarities
import jieba


# 训练样本
raw_documents = [
    '0南京江心洲污泥偷排”等污泥偷排或处置不当而造成的污染问题，不断被媒体曝光',
    '1面对美国金融危机冲击与国内经济增速下滑形势，中国政府在2008年11月初快速推出“4万亿”投资十项措施',
    '2全国大面积出现的雾霾，使解决我国环境质量恶化问题的紧迫性得到全社会的广泛关注',
    '3大约是1962年的夏天吧，潘文突然出现在我们居住的安宁巷中，她旁边走着40号王孃孃家的大儿子，一看就知道，他们是一对恋人。那时候，潘文梳着一条长长的独辫',
    '4坐落在美国科罗拉多州的小镇蒙特苏马有一座4200平方英尺(约合390平方米)的房子，该建筑外表上与普通民居毫无区别，但其内在构造却别有洞天',
    '5据英国《每日邮报》报道，美国威斯康辛州的非营利组织“占领麦迪逊建筑公司”(OMBuild)在华盛顿和俄勒冈州打造了99平方英尺(约9平方米)的迷你房屋',
    '6长沙市公安局官方微博@长沙警事发布消息称，3月14日上午10时15分许，长沙市开福区伍家岭沙湖桥菜市场内，两名摊贩因纠纷引发互殴，其中一人被对方砍死',
    '7乌克兰克里米亚就留在乌克兰还是加入俄罗斯举行全民公投，全部选票的统计结果表明，96.6%的选民赞成克里米亚加入俄罗斯，但未获得乌克兰和国际社会的普遍承认',
    '8京津冀的大气污染，造成了巨大的综合负面效应，显性的是空气污染、水质变差、交通拥堵、食品不安全等，隐性的是各种恶性疾病的患者增加，生存环境越来越差',
    '9 1954年2月19日，苏联最高苏维埃主席团，在“兄弟的乌克兰与俄罗斯结盟300周年之际”通过决议，将俄罗斯联邦的克里米亚州，划归乌克兰加盟共和国',
    '10北京市昌平区一航空训练基地，演练人员身穿训练服，从机舱逃生门滑降到地面',
    '11腾讯入股京东的公告如期而至，与三周前的传闻吻合。毫无疑问，仅仅是传闻阶段的“联姻”，已经改变了京东赴美上市的舆论氛围',
    '12国防部网站消息，3月8日凌晨，马来西亚航空公司MH370航班起飞后与地面失去联系，西安卫星测控中心在第一时间启动应急机制，配合地面搜救人员开展对失联航班的搜索救援行动',
    '13新华社昆明3月2日电，记者从昆明市政府新闻办获悉，昆明“3·01”事件事发现场证据表明，这是一起由新疆分裂势力一手策划组织的严重暴力恐怖事件',
    '14在即将召开的全国“两会”上，中国政府将提出2014年GDP增长7.5%左右、CPI通胀率控制在3.5%的目标',
    '15中共中央总书记、国家主席、中央军委主席习近平看望出席全国政协十二届二次会议的委员并参加分组讨论时强调，团结稳定是福，分裂动乱是祸。全国各族人民都要珍惜民族大团结的政治局面，都要坚决反对一切危害各民族大团结的言行'
]

raw_documents = [
    '0南京江心洲污泥偷排”等污泥偷排或处置不当而造成的污染问题，不断被媒体曝光',
]
corpora_documents = []
# 分词处理
for item_text in raw_documents:
    item_seg = list(jieba.cut(item_text))
    corpora_documents.append(item_seg)

# 生成字典和向量语料
dictionary = corpora.Dictionary(corpora_documents)
# print(dictionary)
# dictionary.save('dict.txt') #保存生成的词典
# dictionary=Dictionary.load('dict.txt')#加载

# 通过下面一句得到语料中每一篇文档对应的稀疏向量（这里是bow向量）
corpus = [dictionary.doc2bow(text) for text in corpora_documents]
# 向量的每一个元素代表了一个word在这篇文档中出现的次数
# print(corpus)
# corpora.MmCorpus.serialize('corpuse.mm',corpus)#保存生成的语料
# corpus=corpora.MmCorpus('corpuse.mm')#加载

# corpus是一个返回bow向量的迭代器。下面代码将完成对corpus中出现的每一个特征的IDF值的统计工作
# tfidf_model = models.TfidfModel(corpus)
# corpus_tfidf = tfidf_model[corpus]

similarity = similarities.Similarity('Similarity-bow-index', corpus, num_features=600)
test_data_1 = '心洲污泥偷排”等污泥偷排或处置不当而造成的污染问题，不断被媒体曝光'
test_cut_raw_1 = list(jieba.cut(test_data_1))
test_corpus_1 = dictionary.doc2bow(test_cut_raw_1)
similarity.num_best = 2
print(similarity[test_corpus_1])  # 返回最相似的样本材料,(index_of_document, similarity) tuples
