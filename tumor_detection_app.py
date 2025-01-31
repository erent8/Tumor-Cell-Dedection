import sys
import json
import cv2
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QTableWidget, QTableWidgetItem, QTabWidget,
                           QProgressBar, QMessageBox)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
import io
import os

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Decoder
        self.dec1 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec3 = self.conv_block(128 + 64, 64)
        self.dec4 = nn.Conv2d(64, 1, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder
        d1 = self.dec1(torch.cat([self.up(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d1), e2], dim=1))
        d3 = self.dec3(torch.cat([self.up(d2), e1], dim=1))
        out = torch.sigmoid(self.dec4(d3))
        
        return out

class TumorDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gelişmiş Tümör Tespit Sistemi")
        self.setGeometry(100, 100, 1200, 800)
        
        # Model yükleme
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet().to(self.device)
        try:
            model_path = 'tumor_model.pth'
            if not os.path.exists(model_path):
                raise FileNotFoundError("Model dosyası bulunamadı!")
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        except Exception as e:
            QMessageBox.warning(self, "Uyarı", f"Model yükleme hatası: {str(e)}")
        
        self.setup_ui()
        
    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Sol panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.image_label = QLabel()
        self.image_label.setFixedSize(600, 500)
        self.image_label.setStyleSheet("border: 2px solid #ccc;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        upload_btn = QPushButton("Görüntü Yükle")
        upload_btn.clicked.connect(self.load_image)
        
        analyze_btn = QPushButton("Analiz Et")
        analyze_btn.clicked.connect(self.analyze_image)
        
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(upload_btn)
        left_layout.addWidget(analyze_btn)
        
        # Sağ panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.tabs = QTabWidget()
        
        # Sonuçlar sekmesi
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        self.results_label = QLabel("Sonuçlar:")
        results_layout.addWidget(self.results_label)
        
        # Grafikler sekmesi
        graphs_tab = QWidget()
        graphs_layout = QVBoxLayout(graphs_tab)
        self.graph_label = QLabel()
        graphs_layout.addWidget(self.graph_label)
        
        # Geçmiş sekmesi
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(7)
        self.history_table.setHorizontalHeaderLabels(
            ["Tarih", "Tümör Sayısı", "Boyut", "Güven Skoru", "Risk Seviyesi", "Öneriler", "Dosya"])
        history_layout.addWidget(self.history_table)
        
        self.tabs.addTab(results_tab, "Sonuçlar")
        self.tabs.addTab(graphs_tab, "Grafikler")
        self.tabs.addTab(history_tab, "Geçmiş")
        
        right_layout.addWidget(self.tabs)
        
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)
        
        self.current_image = None
        self.load_history()
        
    def preprocess_image(self, image):
        try:
            # OpenCV BGR'dan RGB'ye dönüştür
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # NumPy array'i PIL Image'a dönüştür
            pil_image = Image.fromarray(image)
            
            # Dönüşüm işlemleri
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Dönüşümü uygula
            tensor = transform(pil_image)
            
            # Batch boyutu ekle
            tensor = tensor.unsqueeze(0)
            
            return tensor.to(self.device)
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Görüntü işleme hatası: {str(e)}")
            return None
        
    def analyze_image(self):
        if self.current_image is None:
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        try:
            # Görüntü ön işleme
            input_tensor = self.preprocess_image(self.current_image)
            
            self.progress_bar.setValue(30)
            
            # Model tahmini
            with torch.no_grad():
                mask_pred = self.model(input_tensor)
                mask_pred = (mask_pred > 0.5).float()
                
            self.progress_bar.setValue(60)
            
            # Sonuç görüntüsünü hazırla
            mask = mask_pred[0, 0].cpu().numpy()
            mask = cv2.resize(mask, (self.current_image.shape[1], self.current_image.shape[0]))
            
            result_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # Maske üzerinde tümör bölgelerini bul
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            tumor_data = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum alan filtresi
                    # Tümör bölgesini mavi renkle doldur
                    tumor_mask = np.zeros_like(result_image)
                    cv2.drawContours(tumor_mask, [contour], -1, (0, 255, 255), -1)  # BGR formatında mavi
                    
                    # Yarı saydamlık için alpha blending
                    alpha = 0.4
                    mask_area = (tumor_mask > 0)
                    result_image[mask_area] = cv2.addWeighted(result_image[mask_area], 1-alpha, 
                                                            tumor_mask[mask_area], alpha, 0)
                    
                    # Tümör özellikleri
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    confidence = float(np.mean(mask[mask_uint8 > 0]))
                    
                    tumor_data.append({
                        'area': area,
                        'circularity': circularity,
                        'confidence': confidence,
                        'contour': contour
                    })
            
            self.progress_bar.setValue(90)
            
            self.display_image(result_image)
            self.save_result(tumor_data)
            self.plot_statistics(tumor_data)
            
            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Hata", f"Analiz sırasında bir hata oluştu: {str(e)}")
    
    def get_risk_color(self, area, circularity, confidence):
        risk_score = 0
        if area > 2000: risk_score += 1
        if circularity < 0.7: risk_score += 1
        if confidence > 0.8: risk_score += 1
        
        if risk_score >= 2:
            return (0, 0, 255)  # Kırmızı - Yüksek risk
        elif risk_score == 1:
            return (0, 255, 255)  # Sarı - Orta risk
        else:
            return (0, 255, 0)  # Yeşil - Düşük risk
    
    # Diğer yardımcı metodlar...
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Görüntü Seç", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.current_image = cv2.imread(file_name)
            self.display_image(self.current_image)
            
    def display_image(self, image):
        if image is None:
            return
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), 
                                    Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)

    def save_result(self, tumor_data):
        if not tumor_data:
            return
            
        result = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tumor_count": len(tumor_data),
            "sizes": [t['area'] for t in tumor_data],
            "confidence_scores": [t['confidence'] for t in tumor_data],
            "risk_levels": self.calculate_risk_levels(tumor_data),
            "recommendations": self.generate_recommendations(tumor_data)
        }
        
        try:
            with open("tumor_history.json", "r") as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []
            
        history.append(result)
        
        with open("tumor_history.json", "w") as f:
            json.dump(history, f)
            
        self.load_history()
        self.update_results_label(result)

    def calculate_risk_levels(self, tumor_data):
        risk_levels = {'high': 0, 'medium': 0, 'low': 0}
        
        for tumor in tumor_data:
            risk_score = 0
            if tumor['area'] > 2000: risk_score += 1
            if tumor['circularity'] < 0.7: risk_score += 1
            if tumor['confidence'] > 0.8: risk_score += 1
            
            if risk_score >= 2:
                risk_levels['high'] += 1
            elif risk_score == 1:
                risk_levels['medium'] += 1
            else:
                risk_levels['low'] += 1
                
        return risk_levels

    def generate_recommendations(self, tumor_data):
        if not tumor_data:
            return "Tümör tespit edilemedi."
            
        high_risk = sum(1 for t in tumor_data if t['confidence'] > 0.8)
        total = len(tumor_data)
        
        recommendations = []
        
        if high_risk > 0:
            recommendations.append("Acil tıbbi konsültasyon önerilir.")
            if high_risk / total > 0.5:
                recommendations.append("Yüksek riskli bulgular çoğunlukta. İleri tetkik gerekli.")
        elif total > 0:
            recommendations.append("Rutin kontrol önerilir.")
            
        return "\n".join(recommendations)

    def plot_statistics(self, tumor_data):
        if not tumor_data:
            return
            
        plt.clf()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Boyut dağılımı
        sizes = [t['area'] for t in tumor_data]
        ax1.hist(sizes, bins=10)
        ax1.set_title('Tümör Boyut Dağılımı')
        
        # Güven skoru dağılımı
        confidences = [t['confidence'] for t in tumor_data]
        ax2.hist(confidences, bins=10)
        ax2.set_title('Güven Skoru Dağılımı')
        
        # Risk dağılımı
        risk_levels = self.calculate_risk_levels(tumor_data)
        ax3.pie([risk_levels['high'], risk_levels['medium'], risk_levels['low']],
                labels=['Yüksek', 'Orta', 'Düşük'],
                colors=['red', 'yellow', 'green'])
        ax3.set_title('Risk Dağılımı')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = QImage.fromData(buf.getvalue())
        self.graph_label.setPixmap(QPixmap.fromImage(image))
        
        plt.close()

    def update_results_label(self, result):
        """Sonuçları ekranda göster"""
        text = f"""
        Analiz Sonuçları:
        -----------------
        Tespit Edilen Tümör Sayısı: {result['tumor_count']}
        
        Tümör Boyutları:
        Minimum: {min(result['sizes']):.1f} piksel²
        Maksimum: {max(result['sizes']):.1f} piksel²
        Ortalama: {sum(result['sizes'])/len(result['sizes']):.1f} piksel²
        
        Güven Skorları:
        Minimum: {min(result['confidence_scores']):.2f}
        Maksimum: {max(result['confidence_scores']):.2f}
        Ortalama: {sum(result['confidence_scores'])/len(result['confidence_scores']):.2f}
        
        Risk Seviyeleri:
        Yüksek Risk: {result['risk_levels']['high']}
        Orta Risk: {result['risk_levels']['medium']}
        Düşük Risk: {result['risk_levels']['low']}
        
        Öneriler:
        {result['recommendations']}
        """
        self.results_label.setText(text)

    def load_history(self):
        """Geçmiş kayıtları yükle"""
        try:
            # Geçmiş kayıtlar dosyası yoksa boş bir liste oluştur
            if not os.path.exists("tumor_history.json"):
                with open("tumor_history.json", "w") as f:
                    json.dump([], f)
                return

            # Geçmiş kayıtları yükle
            with open("tumor_history.json", "r") as f:
                history = json.load(f)
            
            # Tabloyu temizle
            self.history_table.setRowCount(0)
            
            # Kayıtları tabloya ekle
            for i, data in enumerate(history):
                self.history_table.insertRow(i)
                self.history_table.setItem(i, 0, QTableWidgetItem(data.get("date", "")))
                self.history_table.setItem(i, 1, QTableWidgetItem(str(data.get("tumor_count", 0))))
                
                # Boyutları string olarak birleştir
                sizes = data.get("sizes", [])
                size_str = ", ".join([f"{s:.1f}" for s in sizes]) if sizes else ""
                self.history_table.setItem(i, 2, QTableWidgetItem(size_str))
                
                # Güven skorlarını string olarak birleştir
                scores = data.get("confidence_scores", [])
                score_str = ", ".join([f"{s:.2f}" for s in scores]) if scores else ""
                self.history_table.setItem(i, 3, QTableWidgetItem(score_str))
                
                # Risk seviyelerini göster
                risk_levels = data.get("risk_levels", {})
                risk_str = f"Y:{risk_levels.get('high', 0)} O:{risk_levels.get('medium', 0)} D:{risk_levels.get('low', 0)}"
                self.history_table.setItem(i, 4, QTableWidgetItem(risk_str))
                
                # Önerileri göster
                self.history_table.setItem(i, 5, QTableWidgetItem(data.get("recommendations", "")))
                
                # Dosya adını göster
                self.history_table.setItem(i, 6, QTableWidgetItem(data.get("file_name", "")))
                
            # Sütun genişliklerini ayarla
            self.history_table.resizeColumnsToContents()
            
        except Exception as e:
            QMessageBox.warning(self, "Uyarı", f"Geçmiş kayıtlar yüklenirken hata oluştu: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TumorDetectionApp()
    window.show()
    sys.exit(app.exec())
