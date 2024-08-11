import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';

interface GeneralSetting {
  id: string;
  name: string;
  description: string;
  value: any;
  isSelected: boolean;
}

interface Model {
  id: string;
  name: string;
  description: string;
  isSelected: boolean;
}

interface Dataset {
  id: string;
  name: string;
  description: string;
  isSelected: boolean;
}

interface FilterParam {
  presets: any[];
  value: any;
  isSelected: boolean;
}

interface FilterSettings {
  [key: string]: FilterParam;
}

@Component({
  selector: 'nexusai-perfanalyzer-aiperfanalyzerform',
  templateUrl: './aiperfanalyzerform.component.html',
  styleUrls: ['./aiperfanalyzerform.component.css']
})
export class AiperfanalyzerformComponent {
  models: Model[] = [
    { id: 'resnet50', name: 'Resnet 50', description: 'A 50 layered CNN model', isSelected: true },
    { id: 'yolov2', name: 'YOLO V2', description: 'A you only look once model', isSelected: true },
  ];

  generalSettings: GeneralSetting[] = [
    { id: 'resnet50', name: 'Resnet 50', description: 'A 50 layered CNN model', isSelected: true },
    { id: 'yolov2', name: 'YOLO V2', description: 'A you only look once model', isSelected: true },
  ];

  datasets: Dataset[] = [
    { id: 'cbis', name: 'CBIS-DDSM', description: 'Containing mammogram images (grayscale)', isSelected: true },
    { id: 'breakhis', name: 'BreakHis', description: 'Containing microscopic images (rgb)', isSelected: true },
  ];

  filters: FilterSettings = {
    generateRandom: { presets: [], value: false, isSelected: false },
    numberOfRandomFilters: { presets: [], value: 1, isSelected: false },
    numberOfCombinations: { presets: [], value: 1, isSelected: false },
    // Add more filters with presets and default values
  };

  filterForm: FormGroup;

  constructor(private formBuilder: FormBuilder) {
    const formGroup: { [key: string]: any } = {};
    for (const key of Object.keys(this.filters)) {
      formGroup[key] = [{ value: this.filters[key].value, disabled: true }, Validators.required];
    }
    this.filterForm = this.formBuilder.group(formGroup);
  }

  onSubmit(): void {
    // Implement logic to apply filters using this.filterForm.value
    console.log('Applying filters:', this.filterForm.value);
    // You can perform the actual filter logic here
  }
}
