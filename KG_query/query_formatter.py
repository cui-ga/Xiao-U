from typing import Dict, List, Any, Optional


class QueryResultFormatter:
    """查询结果格式化器 - 将原始查询结果转换为友好的展示格式"""

    @staticmethod
    def format_symptom_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化症状查询结果 - 针对数组类型的symptom"""
        if not result:
            return {"answer": "未找到该疾病的症状信息"}

        disease_name = result.get('disease_name', '未知疾病')
        symptoms = result.get('symptoms', [])  # 现在是字符串数组

        if not symptoms:
            return {
                "answer": f"{disease_name}的症状信息暂时没有记录",
                "details": result
            }

        # 症状是字符串数组
        if isinstance(symptoms, list):
            symptom_list = symptoms
        else:
            symptom_list = [str(symptoms)] if symptoms else []

        symptom_count = len(symptom_list)

        answer = f"{disease_name}的常见症状包括：{', '.join(symptom_list[:10])}"
        if symptom_count > 10:
            answer += f"等{symptom_count}个症状"

        return {
            "answer": answer,
            "details": {
                "disease": disease_name,
                "symptoms": symptom_list,
                "total_count": symptom_count
            }
        }

    @staticmethod
    def format_department_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化科室查询结果 - 针对数组类型的cure_department"""
        if not result:
            return {"answer": "未找到该疾病对应的科室信息"}

        disease_name = result.get('disease_name', '未知疾病')
        departments = result.get('departments', [])

        if not departments:
            return {
                "answer": f"{disease_name}的科室信息暂时没有记录",
                "details": result
            }

        # 科室是字符串数组
        if isinstance(departments, list):
            dept_list = departments
        else:
            dept_list = [str(departments)] if departments else []

        dept_count = len(dept_list)

        answer = f"{disease_name}通常就诊于：{', '.join(dept_list)}"

        return {
            "answer": answer,
            "details": {
                "disease": disease_name,
                "departments": dept_list,
                "total_count": dept_count
            }
        }

    @staticmethod
    def format_treatment_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化治疗方法查询结果 - 针对数组类型的cure_way"""
        if not result:
            return {"answer": "未找到该疾病的治疗方法信息"}

        disease_name = result.get('disease_name', '未知疾病')
        treatments = result.get('treatments', [])

        if not treatments:
            return {
                "answer": f"{disease_name}的治疗方法信息暂时没有记录",
                "details": result
            }

        # 治疗方法是字符串数组
        if isinstance(treatments, list):
            treatment_list = treatments
        else:
            treatment_list = [str(treatments)] if treatments else []

        treatment_count = len(treatment_list)

        answer = f"{disease_name}的治疗方法包括：{', '.join(treatment_list)}"

        return {
            "answer": answer,
            "details": {
                "disease": disease_name,
                "treatments": treatment_list,
                "total_count": treatment_count
            }
        }

    @staticmethod
    def format_cure_rate_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化治愈率查询结果"""
        if not result:
            return {"answer": "未找到该疾病的治愈率信息"}

        disease_name = result.get('disease_name', '未知疾病')
        cure_rate = result.get('cure_rate', '')

        if not cure_rate:
            return {
                "answer": f"{disease_name}的治愈率信息暂时没有记录",
                "details": result
            }

        answer = f"{disease_name}的治愈率：{cure_rate}"

        return {
            "answer": answer,
            "details": {
                "disease": disease_name,
                "cure_rate": cure_rate
            }
        }

    @staticmethod
    def format_cure_time_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化治疗时间查询结果"""
        if not result:
            return {"answer": "未找到该疾病的治疗时间信息"}

        disease_name = result.get('disease_name', '未知疾病')
        cure_time = result.get('cure_time', '')

        if not cure_time:
            return {
                "answer": f"{disease_name}的治疗时间信息暂时没有记录",
                "details": result
            }

        answer = f"{disease_name}的治疗时间：{cure_time}"

        return {
            "answer": answer,
            "details": {
                "disease": disease_name,
                "cure_time": cure_time
            }
        }

    @staticmethod
    def format_cause_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化病因查询结果"""
        if not result:
            return {"answer": "未找到该疾病的病因信息"}

        disease_name = result.get('disease_name', '未知疾病')
        cause = result.get('cause', '')

        if not cause:
            return {
                "answer": f"{disease_name}的病因信息暂时没有记录",
                "details": result
            }

        # 截取前200个字符
        cause_preview = cause[:200] + "..." if len(cause) > 200 else cause
        answer = f"{disease_name}的病因：{cause_preview}"

        return {
            "answer": answer,
            "details": {
                "disease": disease_name,
                "cause": cause
            }
        }

    @staticmethod
    def format_prevent_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化预防措施查询结果"""
        if not result:
            return {"answer": "未找到该疾病的预防措施信息"}

        disease_name = result.get('disease_name', '未知疾病')
        prevent = result.get('prevent', '')

        if not prevent:
            return {
                "answer": f"{disease_name}的预防措施信息暂时没有记录",
                "details": result
            }

        # 截取前200个字符
        prevent_preview = prevent[:200] + "..." if len(prevent) > 200 else prevent
        answer = f"{disease_name}的预防措施：{prevent_preview}"

        return {
            "answer": answer,
            "details": {
                "disease": disease_name,
                "prevent": prevent
            }
        }

    @staticmethod
    def format_check_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化检查项目查询结果 - 针对关系查询结果"""
        if not result:
            return {"answer": "未找到该疾病的相关检查信息"}

        disease_name = result.get('disease_name', '未知疾病')
        checks = result.get('checks', [])
        check_count = result.get('check_count', 0)

        if not checks:
            return {
                "answer": f"{disease_name}的相关检查信息暂时没有记录",
                "details": result
            }

        # 提取检查项目名称
        check_names = [c.get('name', '') for c in checks if c.get('name')]

        answer = f"{disease_name}通常需要进行以下检查：{', '.join(check_names[:5])}"
        if check_count > 5:
            answer += f"等{check_count}项检查"

        return {
            "answer": answer,
            "details": {
                "disease": disease_name,
                "checks": checks,
                "total_count": check_count
            }
        }

    @staticmethod
    def format_definition_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化定义查询结果"""
        if not result:
            return {"answer": "未找到该疾病的定义信息"}

        disease_name = result.get('disease_name', '未知疾病')
        description = result.get('description', '')

        if not description:
            return {
                "answer": f"{disease_name}的定义信息暂时没有记录",
                "details": result
            }

        # 截取前300个字符
        desc_preview = description[:300] + "..." if len(description) > 300 else description
        answer = f"{disease_name}：{desc_preview}"

        return {
            "answer": answer,
            "details": {
                "disease": disease_name,
                "description": description
            }
        }

    @staticmethod
    def format_infectious_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化传染性查询结果"""
        if not result:
            return {"answer": "未找到该疾病的传染性信息"}

        disease_name = result.get('disease_name', '未知疾病')

        # 数据库中无传染性字段
        answer = f"{disease_name}的传染性信息暂时没有记录"

        return {
            "answer": answer,
            "details": {
                "disease": disease_name
            }
        }

    @staticmethod
    def format_taboo_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化禁忌查询结果 - 智能推理版"""
        if not result:
            return {"answer": "未找到该疾病的禁忌信息"}

        disease_name = result.get('disease_name', '未知疾病')

        # 从多个字段中提取可能的禁忌信息
        taboo_keywords = ['禁忌', '避免', '不要', '不宜', '禁止', '忌', '慎用']

        # 检查各个字段中是否包含禁忌相关关键词
        taboo_info = []

        # 从预防措施中找
        prevent = result.get('prevent', '')
        if prevent and any(keyword in prevent for keyword in taboo_keywords):
            taboo_info.append(f"预防措施提示：{prevent[:100]}...")

        # 从治疗方法中找
        treatments = result.get('treatments', [])
        if treatments and isinstance(treatments, list):
            treatment_str = '，'.join(treatments)
            if any(keyword in treatment_str for keyword in taboo_keywords):
                taboo_info.append(f"治疗方法注意事项：{treatment_str}")

        # 从描述中找
        description = result.get('description', '')
        if description and any(keyword in description for keyword in taboo_keywords):
            taboo_info.append(f"疾病描述提示：{description[:100]}...")

        if taboo_info:
            answer = f"{disease_name}的相关禁忌和注意事项：\n" + "\n".join([f"- {info}" for info in taboo_info])
        else:
            answer = f"{disease_name}的禁忌信息暂时没有记录"

        return {
            "answer": answer,
            "details": {
                "disease": disease_name,
                "taboo_sources": taboo_info if taboo_info else "无相关信息"
            }
        }

    @staticmethod
    def format_related_disease_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化相关疾病查询结果"""
        if not result:
            return {"answer": "未找到相关疾病信息"}

        disease_name = result.get('disease_name', '未知疾病')
        related_diseases = result.get('related_diseases', [])
        related_count = result.get('related_count', 0)

        if not related_diseases:
            return {
                "answer": f"{disease_name}的相关疾病信息暂时没有记录",
                "details": result
            }

        # 提取相关疾病名称
        related_names = [d.get('name', '') for d in related_diseases if d.get('name')]

        answer = f"{disease_name}的相关疾病包括：{', '.join(related_names[:5])}"
        if related_count > 5:
            answer += f"等{related_count}种相关疾病"

        return {
            "answer": answer,
            "details": {
                "disease": disease_name,
                "related_diseases": related_diseases,
                "total_count": related_count
            }
        }

    @staticmethod
    def format_by_intent(intent: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """根据意图格式化结果"""
        formatter_map = {
            "临床表现(病症表现)": QueryResultFormatter.format_symptom_result,
            "所属科室": QueryResultFormatter.format_department_result,
            "治疗方法": QueryResultFormatter.format_treatment_result,
            "化验/体检方案": QueryResultFormatter.format_check_result,
            "定义": QueryResultFormatter.format_definition_result,
            "传染性": QueryResultFormatter.format_infectious_result,
            "治愈率": QueryResultFormatter.format_cure_rate_result,
            "治疗时间": QueryResultFormatter.format_cure_time_result,
            "病因": QueryResultFormatter.format_cause_result,
            "相关病症": QueryResultFormatter.format_related_disease_result,
            "禁忌": QueryResultFormatter.format_taboo_result,
            "预防": QueryResultFormatter.format_prevent_result,
            "其他": lambda r: QueryResultFormatter._format_generic_result(r, "基本信息")
        }

        formatter = formatter_map.get(intent, QueryResultFormatter._format_generic_result)
        return formatter(result)

    @staticmethod
    def _format_generic_result(result: Dict[str, Any], field_name: str = "信息") -> Dict[str, Any]:
        """通用格式化方法"""
        if not result:
            return {"answer": f"未找到相关{field_name}"}

        disease_name = result.get('disease_name', '未知疾病')

        # 查找第一个非disease_name的字段
        for key, value in result.items():
            if key != 'disease_name' and value:
                if isinstance(value, (str, list, dict)) and value:
                    if isinstance(value, str) and value.strip():
                        value_preview = value[:200] + "..." if len(value) > 200 else value
                        return {
                            "answer": f"{disease_name}的{field_name}：{value_preview}",
                            "details": result
                        }
                    elif isinstance(value, (list, dict)):
                        return {
                            "answer": f"{disease_name}的{field_name}已找到，详情请查看详细信息",
                            "details": result
                        }

        return {
            "answer": f"{disease_name}的{field_name}暂时没有记录",
            "details": result
        }