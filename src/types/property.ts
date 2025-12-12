export interface PropertyFormData {
  district: string;
  sro_name: string;
  village: string;
  survey_number: string;
  subdivision: string;
}

export interface PropertyFormErrors {
  district?: string;
  sro_name?: string;
  village?: string;
  survey_number?: string;
  subdivision?: string;
}

export type AnalysisVerdict = 'APPROVED' | 'CONDITIONALLY_APPROVED' | 'REJECTED';

export type FindingCategory = 'Mandatory' | 'Optional';

export type FindingSeverity = 'High' | 'Medium' | 'Low';

export interface AnalysisFinding {
  id: string;
  category: FindingCategory;
  severity: FindingSeverity;
  title: string;
  description: string;
  affected_documents: string[];
}

export interface AnalysisResult {
  verdict: AnalysisVerdict;
  confidence_score: number;
  findings: AnalysisFinding[];
  total_documents_analyzed: number;
  summary: string;
}

export interface SROData {
  [district: string]: string[];
}

export interface VillageData {
  [key: string]: string[];
}

export const DISTRICTS = [
  'Chennai',
  'Coimbatore',
  'Madurai',
  'Tiruppur',
  'Erode',
];

export const SRO_BY_DISTRICT: SROData = {
  Chennai: ['Chennai Central', 'Chennai North', 'Chennai South', 'Chennai East'],
  Coimbatore: [
    'Coimbatore City',
    'Coimbatore Rural',
    'Pollachi',
    'Mettupalayam',
  ],
  Madurai: ['Madurai City', 'Madurai Rural', 'Tirunelveli', 'Kanniyakumari'],
  Tiruppur: ['Tiruppur City', 'Tiruppur Rural', 'Udumalaipettai', 'Kangayam'],
  Erode: ['Erode City', 'Erode Rural', 'Bhavani', 'Gobichettipalayam'],
};

export const VILLAGES_BY_SRO: VillageData = {
  'Chennai Central': [
    'Teynampet',
    'Nungambakkam',
    'Alwarpet',
    'Mylapore',
    'Besant Nagar',
  ],
  'Chennai North': [
    'Ambattur',
    'Avadi',
    'Villivakkam',
    'Perambur',
    'Thiruvallur',
  ],
  'Chennai South': [
    'Adyar',
    'Besant Nagar',
    'Velachery',
    'Perungudi',
    'T. Nagar',
  ],
  'Chennai East': [
    'Thiruvanmiyur',
    'Kandanchavadi',
    'Ekkattuthangal',
    'Selaiyur',
    'Palavakkam',
  ],
  'Coimbatore City': [
    'Sundarapuram',
    'Thudiyalur',
    'Vilankurichi',
    'Thadagam',
    'Karumanthurai',
  ],
  'Coimbatore Rural': [
    'Kovai Pudur',
    'Madukkarai',
    'Perur',
    'Mettuppalayam',
    'Orathanadu',
  ],
  Pollachi: ['Andipalayam', 'Lalkudi', 'Parambikulam', 'Udumalaipettai'],
  Mettupalayam: ['Arachalur', 'Coonoor', 'Kunnamkulam', 'Mettupalayam'],
  'Madurai City': [
    'Madurai West',
    'Madurai East',
    'Madurai Central',
    'Arumugha Nagar',
  ],
  'Madurai Rural': [
    'Tirumangalam',
    'Melur',
    'Usilampatti',
    'Peraiyur',
    'Vadipatti',
  ],
  Tirunelveli: [
    'Tirunelveli Town',
    'Radhapuram',
    'Nanguneri',
    'Cheranmahadevi',
  ],
  Kanniyakumari: [
    'Nagercoil',
    'Padmanabhapuram',
    'Colachel',
    'Manapadu',
  ],
  'Tiruppur City': [
    'Tiruppur Town',
    'Avinashi',
    'Noyyal',
    'Kangayam',
    'Palladam',
  ],
  'Tiruppur Rural': [
    'Pongalur',
    'Udumalaipettai',
    'Dharapuram',
    'Madathukulam',
  ],
  Udumalaipettai: ['Tharanampadi', 'Veerasinghpuram', 'Udumalaipettai Town'],
  Kangayam: ['Kangayam Town', 'Bhavanisagar', 'Kangayam Rural'],
  'Erode City': [
    'Erode Town',
    'Bhavani',
    'Gobichettipalayam',
    'Nambiyur',
    'Anthiyur',
  ],
  'Erode Rural': [
    'Kodumudi',
    'Modakurichi',
    'Perundurai',
    'Thalavady',
  ],
  Bhavani: ['Bhavani Town', 'Bhavanisagar', 'Karanthapura'],
  Gobichettipalayam: [
    'Gobichettipalayam Town',
    'Chithode',
    'Chennimalai',
  ],
};
