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
        # cause_preview = cause[:200] + "..." if len(cause) > 200 else cause
        answer = f"{disease_name}的病因：{cause}"

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
        # prevent_preview = prevent[:200] + "..." if len(prevent) > 200 else prevent
        answer = f"{disease_name}的预防措施：{prevent}"

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
        # desc_preview = description[:300] + "..." if len(description) > 300 else description
        answer = f"{disease_name}：{description}"

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
        """格式化禁忌查询结果 - 直接禁忌字段 + 文本注意事项补充"""
        if not result:
            return {"answer": "未找到该疾病的禁忌信息"}

        disease_name = result.get('disease_name', '未知疾病')

        taboo_info = []

        # 1. 优先读取知识图谱直接返回的禁忌/忌食字段
        direct_taboo_foods = (
                result.get("taboo_foods")
                or result.get("forbidden_foods")
                or result.get("not_eat")
                or result.get("忌食食物")
                or result.get("禁忌")
                or []
        )

        if direct_taboo_foods:
            if isinstance(direct_taboo_foods, list):
                foods = [str(item) for item in direct_taboo_foods if item]
            else:
                foods = [str(direct_taboo_foods)]

            if foods:
                taboo_info.append(f"忌食或不建议食用：{', '.join(foods[:10])}")
                if len(foods) > 10:
                    taboo_info[-1] += f"等{len(foods)}项"

        # 2. 再从预防措施、治疗方法、疾病描述中补充禁忌/注意事项
        taboo_keywords = ['禁忌', '避免', '不要', '不宜', '禁止', '忌', '慎用']

        prevent = result.get('prevent', '')
        if prevent and any(keyword in prevent for keyword in taboo_keywords):
            taboo_info.append(f"预防措施提示：{prevent[:100]}...")

        treatments = result.get('treatments', [])
        if treatments and isinstance(treatments, list):
            treatment_str = '，'.join([str(t) for t in treatments if t])
            if any(keyword in treatment_str for keyword in taboo_keywords):
                taboo_info.append(f"治疗方法注意事项：{treatment_str}")

        description = result.get('description', '')
        if description and any(keyword in description for keyword in taboo_keywords):
            taboo_info.append(f"疾病描述提示：{description[:100]}...")

        if taboo_info:
            answer = f"{disease_name}的相关禁忌和注意事项：\n" + "\n".join(
                [f"- {info}" for info in taboo_info]
            )
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
    def normalize_result_keys(result: Dict[str, Any]) -> Dict[str, Any]:
        """字段适配层：将Cypher查询返回的中文字段名统一转换为格式化器使用的英文字段名"""
        if not result or not isinstance(result, dict):
            return result

        key_map = {
            "疾病名称": "disease_name",
            "疾病": "disease_name",
            "名称": "disease_name",
            "症状": "symptoms",
            "症状列表": "symptoms",
            "治疗科室": "departments",
            "科室": "departments",
            "治疗方法": "treatments",
            "疾病描述": "description",
            "描述": "description",
            "传染性": "infectious",
            "治愈率": "cure_rate",
            "治疗周期": "cure_time",
            "治疗时间": "cure_time",
            "病因": "cause",
            "预防措施": "prevent",
            "预防": "prevent",
            "相关检查": "checks",
            "检查项目": "checks",
            "并发症": "related_diseases",
            "相关疾病": "related_diseases",
            "忌食食物": "taboo_foods",
            "忌食": "taboo_foods",
            "宜食食物": "recommended_foods",
            "宜食": "recommended_foods",
            "推荐药品": "recommended_drugs",
            "药品名称": "drug_name",
            "生产厂家": "manufacturer",
            "食物名称": "food_name",
            "菜谱名称": "recipe_name",
            "科室名称": "department_name",
            "药企名称": "company_name",
            "症状名称": "symptom_name",
            "实体类型": "entity_type",
            "属性": "properties",
        }

        normalized = dict(result)

        for old_key, new_key in key_map.items():
            if old_key in result:
                old_value = result.get(old_key)
                if new_key not in normalized or normalized.get(new_key) in (None, "", []):
                    normalized[new_key] = old_value

        # 适配检查结果：format_check_result 期望 checks 为 [{"name": "检查名"}] 结构
        if "checks" in normalized:
            checks = normalized.get("checks")
            if isinstance(checks, list):
                normalized["checks"] = [
                    item if isinstance(item, dict) else {"name": item}
                    for item in checks
                    if item
                ]
            elif checks:
                normalized["checks"] = [{"name": checks}]
            else:
                normalized["checks"] = []
            normalized.setdefault("check_count", len(normalized["checks"]))

        # 适配相关疾病结果：format_related_disease_result 期望 related_diseases 为 [{"name": "疾病名"}] 结构
        if "related_diseases" in normalized:
            related_diseases = normalized.get("related_diseases")
            if isinstance(related_diseases, list):
                normalized["related_diseases"] = [
                    item if isinstance(item, dict) else {"name": item}
                    for item in related_diseases
                    if item
                ]
            elif related_diseases:
                normalized["related_diseases"] = [{"name": related_diseases}]
            else:
                normalized["related_diseases"] = []
            normalized.setdefault("related_count", len(normalized["related_diseases"]))

        # 适配禁忌结果：如果查询模板返回的是“忌食食物”，则转成 format_taboo_result 能识别的 prevent 提示
        taboo_foods = normalized.get("taboo_foods")
        if taboo_foods:
            if isinstance(taboo_foods, list):
                taboo_text = "、".join(str(item) for item in taboo_foods if item)
            else:
                taboo_text = str(taboo_foods)

            if taboo_text:
                taboo_hint = f"忌食食物：{taboo_text}"
                if normalized.get("prevent"):
                    normalized["prevent"] = f"{normalized['prevent']}；{taboo_hint}"
                else:
                    normalized["prevent"] = taboo_hint

        return normalized

    @staticmethod
    def format_by_intent(intent: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """根据意图格式化结果"""
        result = QueryResultFormatter.normalize_result_keys(result)

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
