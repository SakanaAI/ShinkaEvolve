import random
from typing import List, Dict

class OCRDataGenerator:
    """OCR誤りを含むテストデータを生成（シード固定で再現性確保）"""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        
        # よくあるOCR誤読パターン
        self.noise_map = {
            '0': ['O', 'D', 'Q', '()'],
            '1': ['l', 'I', '|', ']', '7'],
            '2': ['Z', 'z'],
            '5': ['S', 's'],
            '8': ['B', '&', 'S'],
            'B': ['8', '13'],
            '-': ['_', '=', '~', ''],
            '/': ['|', 'l', '1'],
            '.': [',', ' '],
            ' ': ['', '.', '_']
        }
        self.prefectures = ["Tokyo", "Osaka", "Kyoto", "Aichi", "Fukuoka"]
        self.cities = ["Shibuya", "Minato", "Chiyoda", "Hakata", "Nagoya"]

    def _inject_noise(self, text: str, noise_prob: float = 0.3) -> str:
        """文字列に確率的にOCRノイズを混入させる"""
        noisy_text = ""
        for char in text:
            if char in self.noise_map and self.rng.random() < noise_prob:
                noisy_text += self.rng.choice(self.noise_map[char])
            else:
                noisy_text += char
        return noisy_text

    def generate_batch(self, batch_size: int = 20) -> List[Dict]:
        """テストデータのバッチを生成"""
        dataset = []
        
        # 1. 電話番号 (090-xxxx-xxxx)
        for _ in range(batch_size // 3):
            truth = f"0{self.rng.randint(70, 90)}-{self.rng.randint(1000, 9999)}-{self.rng.randint(1000, 9999)}"
            dataset.append({
                "type": "phone",
                "ground_truth": truth,
                "input": self._inject_noise(truth, 0.4)
            })

        # 2. 金額 (¥1,234,567)
        for _ in range(batch_size // 3):
            amount = self.rng.randint(1000, 10000000)
            truth = f"¥{amount:,}"
            dataset.append({
                "type": "amount",
                "ground_truth": truth,
                "input": self._inject_noise(truth, 0.3)
            })

        # 3. 住所 (Tokyo-to, Minato-ku...)
        for _ in range(batch_size - len(dataset)):
            pref = self.rng.choice(self.prefectures)
            city = self.rng.choice(self.cities)
            block = f"{self.rng.randint(1,9)}-{self.rng.randint(1,20)}-{self.rng.randint(1,30)}"
            truth = f"{pref}-to, {city}-ku, {block}"
            dataset.append({
                "type": "address",
                "ground_truth": truth,
                "input": self._inject_noise(truth, 0.2)
            })
            
        return dataset