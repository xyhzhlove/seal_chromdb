#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   chroma.py
@Time    :   2024/07/27 08:26:31
@Author  :   许永辉 
@Version :   1.0
@Desc    :   chroma向量数据库封装
'''
import chromadb
import math
import uuid
from typing import List
from .utils import tools, message_format
from .configs import model_config as mcfg
import traceback


class MyMilvus:

    def __init__(self, db_file_path, embeddings):
        # 初始化chroma实例
        self.client = chromadb.PersistentClient(path=db_file_path)
        # 向量化
        self.embeddings = embeddings
        # 向量维度(chroma暂不需要设置向量维度，保证入库的向量同维度即可)
        # self.vector_dim = mcfg.VECTOR_DIM
        # 当前选择的空间
        self.collection = None

    def create_collection(self, collection_name):
        """_创建集合_
        Args:
            collection_name (_str_): _集合(空间名称)_
        Returns:
            _None_: _None_
        """
        if not self.check_collection_exist(collection_name):
            collection = self.client.create_collection(
                name=collection_name, metadata={"hnsw:space": "ip"})
            return collection
        else:
            raise Exception(
                f"Failed create collection in Milvus: {collection_name} has exist"
            )

    def set_collection(self, collection_name):
        """_设置集合_
        Args:
            collection_name (_str_): _集合(空间名称)_
        Returns:
            _None_: _None_
        """
        if self.check_collection_exist(collection_name):
            self.collection = self.client.get_collection(name=collection_name)
            print(f'{collection_name}设置成功')
            return
        else:
            raise Exception(f'{collection_name} has not exsit')

    def create_index(self, collection_name):
        """_创建索引_
        chroma会自动创建索引,所以这里不需要创建索引的方法(预留)
        Args:
            collection_name (_str_): _集合空间名称_
        Returns:
            _None_: _None_
        """
        pass

    def load_collection(self, collection_name):
        """_加载集合_

        Args:
            collection_name (_str_): _集合(空间名称)_
        Returns:
            _None_: _None_
        """
        self.set_collection(collection_name)

    def check_collection_exist(self, collection_name):
        """_检查集合是否存在_

        Args:
            collection_name (_str_): _集合(空间名称)_
        Returns:
            _boolean_: _判断集合(空间是否存在)_
        """
        # 检查集合是否存在
        try:
            collection = self.client.get_collection(name=collection_name)
            return collection is not None
        except Exception as e:
            print(f"Error checking collection existence: {e}")
            return False

    def delete_milvus_table(self, collection_name):
        """_删除集合(空间)_
        Args:
            collection_name (_str_): _集合(空间名称)_
        Returns:
            _str_: _被删除的集合(空间名称)_
        """
        try:
            if self.check_collection_exist(collection_name):
                self.client.delete_collection(collection_name)
                print(f'{collection_name} has delete')
                return collection_name
            else:
                raise Exception(f'{collection_name}！！ has not exsit')
        except Exception as e:
            print(traceback.format_exc())
            return 0

    def query_by_file_list(self, collection_name, file_name_list):
        """_根据文件名拿出该空间内该文件列表中每个文件名的所有片段的向量id_
        Args:
            collection_name (_str_): _集合(空间名称)_
            file_name_list (_type_): __文件列表__
        Returns:
            _list_: _查找到的向量id的列表_
        """
        try:
            result = []
            for file_name in file_name_list:
                result_item = self.query_by_file(collection_name, file_name)
                # 合并列表数据
                result += result_item
            return result
        except Exception as e:
            print(traceback.format_exc())

    def query_by_file(self, collection_name, file_name):
        """_给一个文件名,返回向量数据库中该文件名的所有片段向量id_
        class GetResult(TypedDict):
            ids: List[ID]
            embeddings: Optional[List[Embedding]]
            documents: Optional[List[Document]]
            uris: Optional[URIs]
            data: Optional[Loadable]
            metadatas: Optional[List[Metadata]]
            included: Include
        Args:
            collection_name (_str_): _集合空间名称_
            file_name (_str_): _文件名_
        Returns:
            _list_: _id列表_  
        List[dict(GetResult())]
        """
        try:
            self.load_collection(collection_name)
            return self.collection.get(where={"file": {
                "$eq": file_name
            }})['ids']
        except Exception as e:
            print(traceback.format_exc())

    def delete_document_milvus(self, collection_name, file_name):
        """_删除某个空间内某个文件名的所有向量_
        Args:
            collection_name (_str_): _空间名(集合名)_
            file_name (_str_): _指定的文件名_
        Returns:
            _boolean_: _是否删除成功_
        """
        try:
            self.load_collection(collection_name)
            self.collection.delete(where={"file": {"$eq": file_name}})
        # 表示删除空间中metadatas中file为 file_name的文档项
        except Exception as e:
            print(traceback.format_exc())

    def add_document(self, docs: List[message_format.DocumentFormat],
                     collection_name, file_post_url, send_msg):
        """_添加文章片段到集合(空间)中_
        Args:
            docs (_List[DocumentFormat]_): _文章分割后的片段列表_
            collection_name (_str_): _集合(空间名称)_
            file_post_url(_str_):_回调函数请求的服务地址_
            send_msg(dict):_要发送的消息_
        Returns:
            _None_: _None_
        """
        try:
            self.load_collection(collection_name)
            if self.collection:
                ids_list = []
                sentence_list = []
                metadatas_list = []
                for item in docs:
                    embedding_docs_item: dict = {
                        "id": str(uuid.uuid4()),
                        "sentence": item.sentence,
                        "metadatas": {
                            "complete_content": item.complete_content,
                            "is_title": int(item.is_title),
                            "is_head": int(item.is_head),
                            "level": int(item.level),
                            "outline": str(item.outline),
                            "file": str(item.metadata['source']),
                            "index": int(item.metadata['chunk_num'])
                        }
                    }
                    ids_list.append(embedding_docs_item['id'])
                    sentence_list.append(embedding_docs_item['sentence'])
                    metadatas_list.append(embedding_docs_item['metadatas'])
                num_sentences = len(sentence_list)
                num_batches = (num_sentences + mcfg.MILVUS_INSERT_BATCH -
                               1) // mcfg.MILVUS_INSERT_BATCH
                print(f"num_bathes: {num_batches}")
                # 每个batch的进度值
                single_progress = math.floor(1 /
                                             (num_batches) * 100000) / 100000
                for i in range(num_batches):
                    start_index = i * mcfg.MILVUS_INSERT_BATCH
                    end_index = min((i + 1) * mcfg.MILVUS_INSERT_BATCH,
                                    num_sentences)
                    # import faulthandler
                    # faulthandler.enable()
                    # 在引起崩溃的代码片段前加上该语句，打印出引起崩溃的错误
                    # 考虑减小向量维度 减小批量入库的文档数量

                    self.collection.add(
                        documents=sentence_list[start_index:end_index],
                        embeddings=self.embeddings.encode(
                            sentence_list[start_index:end_index],
                            normalize_embeddings=True),
                        metadatas=metadatas_list[start_index:end_index],
                        ids=ids_list[start_index:end_index])
                    if i != num_batches - 1:
                        send_msg["progress"] += single_progress
                        tools.get_callback_request(file_post_url,
                                                   send_msg=send_msg)
                    else:
                        send_msg["progress"] = 1
                        send_msg["message"] = "导入成功"
                        tools.get_callback_request(file_post_url,
                                                   send_msg=send_msg)
            else:
                raise Exception(f'请先加载{collection_name}空间')
        except Exception as e:
            print(traceback.format_exc())

    def get_context_milvus(self, collection_name, metadatas_list, result_list,
                           context_num):
        """_获取查找的相似文本中该相似文本的上下文(要标题也要正文),这里通过文件名与分割的块(metadatas)进行查找,因为id并不是相邻的_

        Args:
            metadatas_list (_list_): _查找的相似度最高的metadatas元信息列表_
            result_list (_list_): _查找的相似度最高的文本列表_
            context_num (_int_): _上下文数量,拼凑的上下文数量_
            meta_datas_list与result_list结果是一一映射的

        Returns:
            _list_: _拼接好的上下文列表_
        """
        try:
            for index, item in enumerate(metadatas_list):
                file = item['file']
                s_index = int(item['index'])
                # 找到此时分割到的块
                # 找到上下文的块
                add_indexs_list = [
                    s_index + val for val in range(1, context_num + 1)
                ]
                # 找到该文件名的所有结果
                r = self.collection.get(where={
                    "file": {
                        "$eq": file
                    },
                })
                metadatas_new_list = r['metadatas']
                document_list = r['documents']
                document_length = len(document_list)
                # print(document_list)
                # 开始把相邻块的内容拼接上去
                for add_index in add_indexs_list:
                    # 从该文件中所有的数据进行筛选
                    for meta_index, metadata_item in enumerate(
                            metadatas_new_list):
                        if metadata_item[
                                'index'] == add_index and add_index < document_length:
                            # print(meta_index)
                            # print(index)
                            result_list[index]['sentence'] += document_list[
                                meta_index]
                            # 找到此次的块之后就停止内循环
                            break
                # 放入该相似度的所有拼接内容

            return result_list
        except Exception as e:
            print(traceback.format_exc())

    def get_context_content(self, collection_name, metadatas_list, result_list,
                            context_num):
        """_获取查找的相似文本中该相似文本的上下文(只要正文，即level==0的上下文),这里通过文件名与分割的块(metadatas)进行查找,因为id并不是相邻的_

        Args:
            metadatas_list (_list_): _查找的相似度最高的数据元信息列表_
            result_list (_list_): _查找的相似度最高的文本列表_
            context_num (_int_): _上下文数量,拼凑的上下文数量_

        Returns:
            _list_: _拼接好的上下文列表_
        """
        try:
            for index, item in enumerate(metadatas_list):
                file = item['file']
                s_index = int(item['index'])
                # 找到此时分割到的块
                # 找到上下文的块
                add_indexs_list = [
                    s_index + val for val in range(1, context_num + 1)
                ]
                # 找到该文件名的所有结果
                r = self.collection.get(where={
                    "file": {
                        "$eq": file
                    },
                })
                metadatas_new_list = r['metadatas']
                document_list = r['documents']
                document_length = len(document_list)
                # print(document_list)
                # 开始把相邻块的内容拼接上去
                # 是否停止
                stop_flag = 0
                for add_index in add_indexs_list:

                    # 从该文件中所有的数据进行筛选
                    for meta_index, metadata_item in enumerate(
                            metadatas_new_list):
                        # 这里只要level==0(即正文结果)
                        if metadata_item[
                                'index'] == add_index and add_index < document_length:
                            if metadata_item['level'] == 0:
                                result_list[index][
                                    'sentence'] += document_list[meta_index]
                            else:
                                stop_flag += 1

                            # 找到此次的块之后就停止内循环
                            break

                    if stop_flag > 0:
                        # print(stop_flag)
                        break

                # 放入该相似度的所有拼接内容

            return result_list
        except Exception as e:
            print(traceback.format_exc())

    def similarity_query_hybrid_search(self,
                                       collection_name,
                                       query,
                                       limit_num=1):
        """_通过问题文本混合查找相似度(没有过滤条件)_

        Args:
            collection_name (_type_): _集合(空间名称)_
            query (_str_): _问题文本_
            limit_num (int, optional): _返回相似项的条数_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        try:
            if query:
                self.load_collection(collection_name)
                # print(f"query: {query}")
                query_embeddings = self.embeddings.encode(
                    [query], normalize_embeddings=True)
                # print(
                #     f"query_embeddings of length : {len(query_embeddings[0])}")
                result = self.collection.query(
                    query_embeddings=query_embeddings, n_results=limit_num)
                # 这里因为id是不连续的，所以返回metadatas中的file 与 index即可锁定上下文返回
                metadatas_list = result['metadatas'][0]
                # metadatas_list为一个列表，包含着匹配到的每一个相似的数据项的metadatas值
                document_list = result['documents'][0]
                distance_list = result["distances"][0]
                ids_list = result["ids"][0]
                # print(distance_list)
                result_list = []
                for document_index, document_item in enumerate(document_list):
                    result_item = {}
                    result_item['id'] = ids_list[document_index]
                    result_item['sentence'] = document_item
                    result_item['key_sentence'] = document_item
                    result_item['is_title'] = metadatas_list[document_index][
                        'is_title']
                    result_item['is_head'] = metadatas_list[document_index][
                        'is_head']
                    result_item['level'] = metadatas_list[document_index][
                        'level']
                    result_item['outline'] = metadatas_list[document_index][
                        'outline']
                    result_item['index'] = metadatas_list[document_index][
                        'index']
                    result_item['file'] = metadatas_list[document_index][
                        'file']
                    result_item['distance'] = 1 - distance_list[document_index]

                    result_list.append(result_item)
                result_list = self.get_context_milvus(collection_name,
                                                      metadatas_list,
                                                      result_list,
                                                      mcfg.CONTEXT_NUM)
                # 处理特殊字符
                result_list = [
                    item for item in result_list if item['distance'] > 0.4
                ]
                for res in result_list:
                    res["sentence"] = res["sentence"].replace(
                        "\xa0", "").replace("\n", "").strip()
                    res["key_sentence"] = res["key_sentence"].replace(
                        "\xa0", "").replace("\n", "").strip()
                return result_list

            else:
                raise Exception('问题不能为空')
        except Exception as e:
            print(traceback.format_exc())

    def similarity_filter_hybrid_search(self,
                                        collection_name,
                                        query,
                                        filter_expr,
                                        limit_num=1,
                                        context_num=2):
        """_通过问题文本混合查找相似度(有过滤条件)_

        Args:
            collection_name (_str_): _集合(空间名称)_
            query (_str_): _问题_
            filter_expr(_str_):过滤条件，例如(is_title==1)表示过滤掉is_title==1的数据项
            limit_num (int, optional): _int_. Defaults to 1.取相似的前几个问题
            context_num=2 表示要的相邻上下文的数据项数量
        Returns:
            _list_: _表示返回的相关内容的列表_
        """
        # filter_expr举例子:{"is_title": {"$ne": 1}}
        # 表示返回的数据项中需要metadatas中的is_title属性不等于1
        try:
            if query:
                self.load_collection(collection_name)
                query_embeddings = self.embeddings.encode(
                    [query], normalize_embeddings=True)
                result = self.collection.query(
                    query_embeddings=query_embeddings,
                    n_results=limit_num,
                    where=filter_expr)
                # 这里因为id是不连续的，所以返回metadatas中的file 与 index即可锁定上下文返回
                metadatas_list = result['metadatas'][0]
                # metadatas_list为一个列表，包含着匹配到的每一个相似的数据项的metadatas值
                docment_list = result['documents'][0]
                distance_list = result["distances"][0]
                ids_list = result["ids"][0]
                # print(distance_list)
                result_list = []
                for document_index, document_item in enumerate(docment_list):
                    result_item = {}
                    result_item['id'] = ids_list[document_index]
                    result_item['sentence'] = document_item.replace(
                        "\xa0", "").replace("\n", "").strip()
                    result_item['is_title'] = metadatas_list[document_index][
                        'is_title']
                    result_item['is_head'] = metadatas_list[document_index][
                        'is_head']
                    result_item['level'] = metadatas_list[document_index][
                        'level']
                    result_item['outline'] = metadatas_list[document_index][
                        'outline']
                    result_item['index'] = metadatas_list[document_index][
                        'index']
                    result_item['file'] = metadatas_list[document_index][
                        'file']
                    result_item['distance'] = 1 - distance_list[document_index]

                    result_list.append(result_item)

                # document_list为一个列表，包含着匹配到的每一个相似的数据项的document(正文)值
                # print(docment_list)
                if context_num > 1:
                    result_list = self.get_context_content(
                        collection_name, metadatas_list, result_list,
                        context_num)
                result_list = [
                    item for item in result_list if item['distance'] > 0.4
                ]
                return result_list

            else:
                raise Exception('问题不能为空')
        except Exception as e:
            print(traceback.format_exc())
