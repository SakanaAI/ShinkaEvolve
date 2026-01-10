import random
from typing import List, Dict

class OCRDataGenerator:
    """
    OCR誤りを含むテストデータを生成
    
    設計思想:
    - 単純な置換ではなく「文脈依存」の判断が必要
    - 同じ文字が異なる意味を持つケース（例: "O"が数字の0か文字のOか）
    - 曖昧なケースでは構造的検証が必要
    """
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        
        # === OCR誤読パターン（双方向） ===
        # 重要: 同じ誤読でも文脈により正解が異なる
        self.noise_map = {
            '0': ['O', 'D', 'Q', '()'],
            '1': ['l', 'I', '|', ']', '7'],
            '2': ['Z', 'z'],
            '5': ['S', 's'],
            '8': ['B', '&'],
            '-': ['_', '=', '~', '一'],  # 「一」は日本語の漢数字
            '/': ['|', 'l', '1'],
            '.': [',', '・'],
            ' ': ['', '.', '_'],
            ',': ['.', '、'],
        }
        
        # === 日本企業向けデータパターン ===
        self.company_names = [
            "TOYOTA", "SONY", "HONDA", "OLYMPUS", "TOSHIBA",
            "MITSUBISHI", "SUMITOMO", "MITSUI", "CANON", "RICOH"
        ]
        self.prefectures = ["Tokyo", "Osaka", "Kyoto", "Aichi", "Fukuoka"]
        self.cities = ["Shibuya", "Minato", "Chiyoda", "Shinjuku", "Nagoya"]
        
        # 日本語の住所表記（漢字混じり）
        self.japanese_addresses = [
            "東京都渋谷区", "大阪府北区", "京都市左京区", "名古屋市中区"
        ]

    def _inject_noise(self, text: str, noise_prob: float = 0.3) -> str:
        """文字列に確率的にOCRノイズを混入"""
        noisy = ""
        for char in text:
            if char in self.noise_map and self.rng.random() < noise_prob:
                noisy += self.rng.choice(self.noise_map[char])
            else:
                noisy += char
        return noisy

    def _inject_contextual_noise(self, text: str, preserve_alpha: bool = False) -> str:
        """
        文脈を考慮したノイズ注入
        preserve_alpha=True の場合、アルファベットは文字として保持すべき
        """
        noisy = ""
        for i, char in enumerate(text):
            # アルファベットを保持すべき文脈
            if preserve_alpha and char.isalpha():
                # 大文字アルファベットは誤読されうるが、正解は文字のまま
                if char in ['O', 'I', 'S', 'B', 'Z'] and self.rng.random() < 0.3:
                    # 数字に誤読されたが、正解はアルファベット
                    noise_to_char = {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'Z': '2'}
                    noisy += noise_to_char.get(char, char)
                else:
                    noisy += char
            elif char in self.noise_map and self.rng.random() < 0.35:
                noisy += self.rng.choice(self.noise_map[char])
            else:
                noisy += char
        return noisy

    def generate_batch(self, batch_size: int = 20) -> List[Dict]:
        """多様なテストデータを生成"""
        dataset = []
        samples_per_type = max(1, batch_size // 5)
        
        # === 1. 電話番号（数字のみ、Oは0） ===
        for _ in range(samples_per_type):
            truth = f"0{self.rng.randint(70, 90)}-{self.rng.randint(1000, 9999)}-{self.rng.randint(1000, 9999)}"
            dataset.append({
                "type": "phone",
                "ground_truth": truth,
                "input": self._inject_noise(truth, 0.4)
            })

        # === 2. 金額（数字とカンマ） ===
        for _ in range(samples_per_type):
            amount = self.rng.randint(1000, 10000000)
            truth = f"¥{amount:,}"
            dataset.append({
                "type": "amount",
                "ground_truth": truth,
                "input": self._inject_noise(truth, 0.35)
            })

        # === 3. 会社名（アルファベット、Oは文字のO） ===
        # ★重要: ここでは "TOYOTA" の "O" は数字ではない
        for _ in range(samples_per_type):
            company = self.rng.choice(self.company_names)
            # 会社コードを付加（例: TOYOTA-12345）
            code = f"{self.rng.randint(10000, 99999)}"
            truth = f"{company}-{code}"
            # preserve_alpha=True で、アルファベットは保持すべきと明示
            dataset.append({
                "type": "company_code",
                "ground_truth": truth,
                "input": self._inject_contextual_noise(truth, preserve_alpha=True)
            })

        # === 4. 混合形式: 請求書番号（アルファベット+数字） ===
        # 例: "INV-2024-00123" の "I" は文字、"0" は数字
        for _ in range(samples_per_type):
            prefix = self.rng.choice(["INV", "ORD", "PO", "SO", "RCV"])
            year = self.rng.randint(2020, 2026)
            num = self.rng.randint(1, 99999)
            truth = f"{prefix}-{year}-{num:05d}"
            dataset.append({
                "type": "invoice_id",
                "ground_truth": truth,
                "input": self._inject_contextual_noise(truth, preserve_alpha=True)
            })

        # === 5. 住所（英語表記） ===
        for _ in range(batch_size - len(dataset)):
            pref = self.rng.choice(self.prefectures)
            city = self.rng.choice(self.cities)
            block = f"{self.rng.randint(1,9)}-{self.rng.randint(1,20)}-{self.rng.randint(1,30)}"
            truth = f"{pref}-to, {city}-ku, {block}"
            dataset.append({
                "type": "address",
                "ground_truth": truth,
                "input": self._inject_noise(truth, 0.25)
            })

        self.rng.shuffle(dataset)
        return dataset