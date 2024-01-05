import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PerfanalyzerComponent } from './perfanalyzer.component';

describe('PerfanalyzerComponent', () => {
  let component: PerfanalyzerComponent;
  let fixture: ComponentFixture<PerfanalyzerComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PerfanalyzerComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(PerfanalyzerComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
