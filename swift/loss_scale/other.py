# Copyright (c) ModelScope Contributors. All rights reserved.
import math
from typing import List, Optional, Tuple, Dict, Any
from .base import LossScale, ConfigLossScale
from swift.template import ContextType, Messages



class IgnoreEmptyThinkLossScale(ConfigLossScale):
    loss_scale_config = 'ignore_empty_think.json'

class ExpertAnnoLossScale(LossScale):
    """
    仅作用于最后一轮assistant

    1) span 模式：
        kwargs["span_weights"] = [{"start": int, "end": int, "weight": float}, ...]
        - 区间语义：[start, end)（start 含，end 不含），单位为字符下标
        - 不允许重叠（边界相接允许）
        - 任一越界/非法/冲突 -> 回退全 1

    2) char 模式（逐字符权重，优先级更高）：
        kwargs["char_weights"] = [w0, w1, ..., w_{len(text)-1}]
        - 长度必须严格等于 len(context)
        - 每个权重需为有限数且 >= 0
        - 任一非法 -> 回退全 1

    冲突规则：
      - 同时提供 char_weights 与 span_weights -> 优先 char 模式
      - 两者都不提供 -> 回退全 1

    返回：
      (segments, loss_scales)
      segments 为字符串片段列表；loss_scales 为对应片段权重。
    """

    is_binary = False

    def _fallback(self, context: Any) -> Tuple[List[Any], List[float]]:
        """回退全1"""
        return [context], [1.0]

    def _merge_segments_by_weight(
        self, text: str, char_weights: List[float]
    ) -> Tuple[List[str], List[float]]:
        """根据字符级权重数组合并成片段"""
        n = len(text)
        if n == 0 or len(char_weights) != n:
            return self._fallback(text)

        segments: List[str] = []
        scales: List[float] = []

        # 按连续相同权重合并成片段
        start = 0
        current_weight = char_weights[0]

        for i in range(1, n):
            if char_weights[i] != current_weight:
                segments.append(text[start:i])
                scales.append(current_weight)
                start = i
                current_weight = char_weights[i]

        # 添加最后一个片段
        segments.append(text[start:])
        scales.append(current_weight)

        # 安全兜底：确保可重构
        if "".join(segments) != text:
            return self._fallback(text)

        return segments, scales

    # char mode
    def _normalize_char_weights(
        self, weights: Any, n_chars: int
    ) -> Optional[List[float]]:
        """验证并标准化字符级权重"""
        if not isinstance(weights, list) or len(weights) != n_chars or n_chars == 0:
            return None

        normalized: List[float] = []
        for w in weights:
            try:
                fw = float(w)
            except (ValueError, TypeError):
                return None

            if not math.isfinite(fw) or fw < 0.0:
                return None
            normalized.append(fw)

        return normalized

    # span mode
    def _spans_to_char_weights(self, spans: Any, n_chars: int) -> Optional[List[float]]:
        """span_weights -> char_weights（包含解析、合法性检查、重叠检查）"""
        if spans is None:
            return None

        # 允许传单个 dict -> 规范化成 list
        if not isinstance(spans, list):
            spans = [spans]

        if len(spans) == 0 or n_chars == 0:
            return None

        normalized: List[Dict[str, Any]] = []
        for span in spans:
            if not isinstance(span, dict):
                return None
            if not all(k in span for k in ("start", "end", "weight")):
                return None

            try:
                s = int(span["start"])
                e = int(span["end"])
                w = float(span["weight"])
            except (ValueError, TypeError):
                return None

            # 边界与权重合法性
            if not (0 <= s < e <= n_chars):
                return None
            if not math.isfinite(w) or w < 0.0:
                return None

            normalized.append({"start": s, "end": e, "weight": w})

        # 重叠检查（边界相接允许）
        normalized.sort(key=lambda x: (x["start"], x["end"]))
        for i in range(1, len(normalized)):
            if normalized[i]["start"] < normalized[i - 1]["end"]:
                return None

        # 生成 char_weights
        char_weights = [1.0] * n_chars
        for sp in normalized:
            s, e, w = sp["start"], sp["end"], sp["weight"]
            char_weights[s:e] = [w] * (e - s)

        return char_weights

    # 入口
    def get_loss_scale(
        self, context: Any, context_type: ContextType, **kwargs
    ) -> Tuple[List[Any], List[float]]:
        """根据 kwargs 中的权重配置返回片段和权重"""
        # context需为str且类型需为response
        if not isinstance(context, str) or context_type == ContextType.SUFFIX:
            return self._fallback(context)

        text = context
        n = len(text)

        # 获取权重标注
        char_weights_input = kwargs.get("char_weights")
        span_weights_input = kwargs.get("span_weights")

        # 检查模式冲突或无配置
        has_char = char_weights_input is not None
        has_span = span_weights_input is not None

        # char 模式（优先）
        if has_char:
            char_weights = self._normalize_char_weights(char_weights_input, n)
            if char_weights is None:
                return self._fallback(context)
        # span 模式
        elif has_span:
            char_weights = self._spans_to_char_weights(span_weights_input, n)
            if char_weights is None:
                return self._fallback(context)
        else:
            return self._fallback(context)

        return self._merge_segments_by_weight(text, char_weights)


    def __call__(self, context_list: List[str], context_types: List[ContextType], messages: Messages,
                 **kwargs) -> Tuple[List[str], List[float]]:
        """Process the complete conversation context and return contexts with loss scales.

            This method iterates through all context segments and determines the loss scale
            for each based on the context type and base strategy. It handles special cases
            such as explicitly specified loss values in messages and pre-computed loss scales.

            Args:
                context_list: List of context strings or dicts, each representing a segment
                    of the conversation.
                context_types: List of context types corresponding to each context, indicating
                    whether it's a system prompt, user query, assistant response, etc.
                messages: Complete message list containing the conversation history.

            Returns:
                A tuple containing:
                    - List[str]: Processed context list, potentially expanded if contexts
                        are split into multiple parts
                    - List[float]: Loss scale values corresponding one-to-one with the
                        returned context list
        """
        res_context_list = []
        res_loss_scale = []
        i = 0
        # last_user_round = get_last_user_round(messages)
        last_assistant_round = sum(1 for m in messages if m['role'] == 'assistant') - 1
        for context, context_type in zip(context_list, context_types):
            # 有问题，最后一个user之后的所有assistant都会被认为是last_round
            # is_last_round = 2 * i >= last_user_round
            is_last_round = i >= last_assistant_round
            query, loss = None, None
            if context_type == ContextType.RESPONSE:
                query = messages[2 * i]['content']
                # Currently, we only support applying loss/mask to the response part.
                loss = messages[2 * i + 1].get('loss')
                assert context == messages[2 * i + 1]['content']
                i += 1
            if isinstance(context, dict) and 'loss_scale' in context:
                new_context = [[token] for token in context['token_ids']]
                loss_scale = context['loss_scale']
            else:
                if isinstance(context, dict) and 'token_ids' in context:
                    context = context['token_ids']
                is_assistant = context_type in {ContextType.RESPONSE, ContextType.SUFFIX}
                if loss or loss is None and (self.base_strategy == 'all' or
                                             (self.base_strategy == 'default' and is_assistant) or
                                             (self.base_strategy == 'last_round' and is_assistant and is_last_round)):
                    new_context, loss_scale = self.get_loss_scale(context, context_type=context_type, **kwargs)
                else:
                    new_context, loss_scale = [context], [0.]
            res_context_list += new_context
            res_loss_scale += loss_scale
        # The values in loss_scale_list correspond one-to-one with the values in context_list.
        return res_context_list, res_loss_scale
