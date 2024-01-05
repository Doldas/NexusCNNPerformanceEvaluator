import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PerfanalyzerComponent } from './perfanalyzer/perfanalyzer.component';
import { ConsolewindowComponent } from './perfanalyzer/consolewindow/consolewindow.component';
import { ControlpanelComponent } from './perfanalyzer/controlpanel/controlpanel.component';
import { AiperfanalyzerformComponent } from './perfanalyzer/controlpanel/aiperfanalyzerform/aiperfanalyzerform.component';
import { FormsModule } from '@angular/forms';

@NgModule({
  declarations: [PerfanalyzerComponent,ConsolewindowComponent,ControlpanelComponent, AiperfanalyzerformComponent],
  imports: [
    CommonModule,
    FormsModule,
  ],
  exports:[
    PerfanalyzerComponent
  ]
})
export class NexusaiModule { }
