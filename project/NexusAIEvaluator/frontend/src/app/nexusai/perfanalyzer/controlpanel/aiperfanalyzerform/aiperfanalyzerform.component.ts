import { Component } from '@angular/core';

// AI MODELS
interface Model {
  id: string;
  name: string;
  description: string;
  isSelected: boolean;
}

// DATASETS
interface Dataset {
  id: string;
  name: string;
  description: string;
  isSelected: boolean;
}

// IMAGE ENHANCEMENT:
interface FilterParam {
  presets: number[];
  value: number;
}

interface FilterSettings {
  [key: string]: FilterParam;
}

@Component({
  selector: 'nexusai-perfanalyzer-aiperfanalyzerform',
  standalone: false,
  templateUrl: './aiperfanalyzerform.component.html',
  styleUrl: './aiperfanalyzerform.component.css'
})
export class AiperfanalyzerformComponent {
  // AI MODELS
  models: Model[] = [
    { id: 'resnet50', name: 'Resnet 50', description: 'A 50 layered CNN model', isSelected: true },
    { id: 'yolov2', name: 'YOLO V2', description: 'A you only look once model', isSelected: true },
    // Add more models as needed
  ];
  // DATASETS
  datasets: Dataset[] = [
    { id: 'cbis', name: 'CBIS-DDSM', description: 'Containing mammogram images (grayscale)', isSelected: true },
    { id: 'breakhis', name: 'BreakHis', description: 'Containing microscopic images (rgb)', isSelected: true },
    // Add more datasets as needed
  ];
  // IMAGE ENHANCEMENT

  // BACKGROUND REMOVAL
  rollingBallEnabled = false;
  rollingBallParams: FilterSettings = {
    strength: { presets: [25, 50, 75,100], value: 25 },
  };

  // DE BLURRING

  // DE NOISING
}
