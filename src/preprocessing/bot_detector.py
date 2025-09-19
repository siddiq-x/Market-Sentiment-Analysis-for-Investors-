"""
Bot detection and credibility scoring for social media content
"""
import re
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib

@dataclass
class BotScore:
    """Container for bot detection results"""
    is_bot_likely: bool
    confidence: float
    reasons: List[str]
    credibility_adjustment: float

class BotDetector:
    """Detect bots and assign credibility scores to social media content"""

    def __init__(self):
        self.logger = logging.getLogger("bot_detector")

        # Bot detection patterns
        self.suspicious_patterns = {
            'username_patterns': [
                r'^[a-zA-Z]+\d{4,}$',  # Name followed by many digits
                r'^[a-zA-Z]+_[a-zA-Z]+\d+$',  # Name_Name123 pattern
                r'^user\d+$',  # Generic user123
                r'^[a-zA-Z]{1,3}\d{6,}$',  # Short letters + many digits
            ],
            'content_patterns': [
                r'(?:buy|sell|invest)\s+(?:now|today|immediately)',  # Urgency
                r'guaranteed\s+(?:profit|return|gains)',  # Unrealistic promises
                r'(?:\d+%|\d+x)\s+(?:profit|return|gains?)\s+(?:guaranteed|certain)',
                r'click\s+(?:here|link|below)',  # Spam links
                r'(?:dm|message)\s+me\s+for',  # Direct message requests
            ],
            'spam_indicators': [
                r'🚀{2,}',  # Multiple rocket emojis
                r'💎{2,}',  # Multiple diamond emojis
                r'[A-Z]{10,}',  # Excessive caps
                r'(?:!!!|…){3,}',  # Excessive punctuation
            ]
        }

        # Credible source indicators
        self.credible_indicators = {
            'verified_domains': [
                'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com',
                'cnbc.com', 'marketwatch.com', 'yahoo.com', 'google.com'
            ],
            'professional_terms': [
                'analysis', 'research', 'fundamental', 'technical',
                'valuation', 'dcf', 'earnings', 'revenue', 'margin'
            ],
            'institutional_language': [
                'according to', 'reported that', 'announced',
                'disclosed', 'filed', 'sec filing', 'quarterly report'
            ]
        }

    def detect_bot(self,
                   content: str,
                   author_info: Optional[Dict[str, Any]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> BotScore:
        """
        Detect if content is likely from a bot

        Args:
            content: Text content to analyze
            author_info: Information about the author (username, followers, etc.)
            metadata: Additional metadata (timestamps, engagement, etc.)
        """
        reasons = []
        bot_score = 0.0

        # Analyze content patterns
        content_score, content_reasons = self._analyze_content_patterns(content)
        bot_score += content_score
        reasons.extend(content_reasons)

        # Analyze author information
        if author_info:
            author_score, author_reasons = self._analyze_author_patterns(author_info)
            bot_score += author_score
            reasons.extend(author_reasons)

        # Analyze temporal patterns
        if metadata:
            temporal_score, temporal_reasons = self._analyze_temporal_patterns(metadata)
            bot_score += temporal_score
            reasons.extend(temporal_reasons)

        # Analyze engagement patterns
        if metadata:
            engagement_score, engagement_reasons = self._analyze_engagement_patterns(metadata)
            bot_score += engagement_score
            reasons.extend(engagement_reasons)

        # Normalize score (0-1 scale)
        normalized_score = min(1.0, bot_score / 4.0)  # Assuming max 4 categories

        is_bot_likely = normalized_score > 0.6
        confidence = abs(normalized_score - 0.5) * 2  # Distance from neutral

        # Calculate credibility adjustment
        credibility_adjustment = self._calculate_credibility_adjustment(
            normalized_score, content, author_info, metadata
        )

        return BotScore(
            is_bot_likely=is_bot_likely,
            confidence=confidence,
            reasons=reasons,
            credibility_adjustment=credibility_adjustment
        )

    def _analyze_content_patterns(self, content: str) -> Tuple[float, List[str]]:
        """Analyze content for bot-like patterns"""
        score = 0.0
        reasons = []

        if not content:
            return score, reasons

        content_lower = content.lower()

        # Check for suspicious content patterns
        for pattern in self.suspicious_patterns['content_patterns']:
            if re.search(pattern, content_lower):
                score += 0.3
                reasons.append(f"Suspicious content pattern: {pattern}")

        # Check for spam indicators
        for pattern in self.suspicious_patterns['spam_indicators']:
            if re.search(pattern, content):
                score += 0.2
                reasons.append(f"Spam indicator: {pattern}")

        # Check for excessive repetition
        words = content_lower.split()
        if len(words) > 5:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.3:  # More than 30% repetition
                score += 0.4
                reasons.append("Excessive word repetition")

        # Check for unrealistic claims
        unrealistic_patterns = [
            r'\d+%\s+(?:guaranteed|sure|certain)',
            r'(?:never|zero)\s+(?:risk|loss)',
            r'(?:easy|quick)\s+(?:money|profit|cash)'
        ]

        for pattern in unrealistic_patterns:
            if re.search(pattern, content_lower):
                score += 0.3
                reasons.append("Unrealistic financial claims")

        return min(1.0, score), reasons

    def _analyze_author_patterns(self, author_info: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze author information for bot indicators"""
        score = 0.0
        reasons = []

        username = author_info.get('username', '').lower()

        # Check username patterns
        for pattern in self.suspicious_patterns['username_patterns']:
            if re.match(pattern, username):
                score += 0.4
                reasons.append("Suspicious username pattern")
                break

        # Check account age vs activity
        account_age_days = author_info.get('account_age_days', 0)
        post_count = author_info.get('post_count', 0)

        if account_age_days > 0 and post_count > 0:
            posts_per_day = post_count / account_age_days
            if posts_per_day > 50:  # More than 50 posts per day
                score += 0.5
                reasons.append("Excessive posting frequency")
            elif posts_per_day > 20:
                score += 0.3
                reasons.append("High posting frequency")

        # Check follower to following ratio
        followers = author_info.get('followers_count', 0)
        following = author_info.get('following_count', 0)

        if following > 0:
            ratio = followers / following
            if ratio < 0.1 and following > 1000:  # Following many, few followers
                score += 0.3
                reasons.append("Suspicious follower/following ratio")

        # Check profile completeness indicators
        if author_info.get('default_profile_image', False):
            score += 0.2
            reasons.append("Default profile image")

        if not author_info.get('bio', '').strip():
            score += 0.1
            reasons.append("Empty profile bio")

        return min(1.0, score), reasons

    def _analyze_temporal_patterns(self, metadata: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze temporal posting patterns"""
        score = 0.0
        reasons = []

        # Check for rapid-fire posting
        recent_posts = metadata.get('recent_post_timestamps', [])
        if len(recent_posts) >= 3:
            timestamps = [datetime.fromisoformat(ts) if isinstance(ts, str) else ts
                        for ts in recent_posts]
            timestamps.sort()

            # Check for posts within very short intervals
            short_intervals = 0
            for i in range(1, len(timestamps)):
                diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                if diff < 60:  # Less than 1 minute apart
                    short_intervals += 1

            if short_intervals >= 2:
                score += 0.4
                reasons.append("Rapid-fire posting pattern")

        # Check for posting at unusual hours consistently
        post_hours = metadata.get('typical_post_hours', [])
        if post_hours:
            night_posts = sum(1 for hour in post_hours if 2 <= hour <= 5)
            if night_posts / len(post_hours) > 0.7:  # More than 70% night posts
                score += 0.2
                reasons.append("Unusual posting hours")

        return min(1.0, score), reasons

    def _analyze_engagement_patterns(self, metadata: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze engagement patterns for bot indicators"""
        score = 0.0
        reasons = []

        # Check engagement ratios
        likes = metadata.get('like_count', 0)
        retweets = metadata.get('retweet_count', 0)
        replies = metadata.get('reply_count', 0)

        total_engagement = likes + retweets + replies

        if total_engagement > 0:
            # Unusual engagement patterns
            if retweets > likes * 2:  # More retweets than likes (unusual)
                score += 0.2
                reasons.append("Unusual retweet/like ratio")

            if replies == 0 and total_engagement > 100:  # High engagement but no replies
                score += 0.1
                reasons.append("No replies despite high engagement")

        # Check for coordinated activity indicators
        if metadata.get('coordinated_activity_score', 0) > 0.7:
            score += 0.5
            reasons.append("Potential coordinated activity")

        return min(1.0, score), reasons

    def _calculate_credibility_adjustment(self,
                                        bot_score: float,
                                        content: str,
                                        author_info: Optional[Dict[str, Any]],
                                        metadata: Optional[Dict[str, Any]]) -> float:
        """Calculate credibility adjustment based on bot score and other factors"""
        base_adjustment = 1.0 - bot_score  # Higher bot score = lower credibility

        # Positive adjustments for credible indicators
        credibility_boost = 0.0

        if content:
            content_lower = content.lower()

            # Check for professional language
            professional_terms = sum(1 for term in self.credible_indicators['professional_terms']
                                   if term in content_lower)
            if professional_terms >= 2:
                credibility_boost += 0.1

            # Check for institutional language
            institutional_terms = sum(1 for term in self.credible_indicators['institutional_language']
                                    if term in content_lower)
            if institutional_terms >= 1:
                credibility_boost += 0.15

        # Check for verified account or credible domain
        if author_info:
            if author_info.get('verified', False):
                credibility_boost += 0.2

            if author_info.get('follower_count', 0) > 10000:
                credibility_boost += 0.1

        if metadata:
            source_domain = metadata.get('source_domain', '').lower()
            if any(domain in source_domain for domain in self.credible_indicators['verified_domains']):
                credibility_boost += 0.3

        # Final credibility adjustment (0.0 to 1.0)
        final_adjustment = min(1.0, max(0.1, base_adjustment + credibility_boost))

        return final_adjustment

    def batch_detect(self,
                    contents: List[str],
                    author_infos: Optional[List[Dict[str, Any]]] = None,
                    metadatas: Optional[List[Dict[str, Any]]] = None) -> List[BotScore]:
        """Detect bots in batch"""
        results = []

        for i, content in enumerate(contents):
            author_info = author_infos[i] if author_infos and i < len(author_infos) else None
            metadata = metadatas[i] if metadatas and i < len(metadatas) else None

            result = self.detect_bot(content, author_info, metadata)
            results.append(result)

        return results

    def get_detection_stats(self, bot_scores: List[BotScore]) -> Dict[str, Any]:
        """Get statistics about bot detection results"""
        if not bot_scores:
            return {}

        bot_count = sum(1 for score in bot_scores if score.is_bot_likely)
        total_count = len(bot_scores)

        avg_confidence = sum(score.confidence for score in bot_scores) / total_count
        avg_credibility = sum(score.credibility_adjustment for score in bot_scores) / total_count

        # Most common reasons
        all_reasons = []
        for score in bot_scores:
            all_reasons.extend(score.reasons)

        reason_counts = {}
        for reason in all_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_analyzed": total_count,
            "likely_bots": bot_count,
            "bot_percentage": (bot_count / total_count) * 100,
            "average_confidence": avg_confidence,
            "average_credibility_adjustment": avg_credibility,
            "top_bot_indicators": top_reasons
        }
