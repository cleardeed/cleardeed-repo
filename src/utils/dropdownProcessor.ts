import dropdownDataRaw from '../data/dropdowns.json';

export interface DropdownOption {
  name: string;
  value: string;
}

export interface CascadingDropdownData {
  zones: DropdownOption[];
  districtsByZone: Map<string, DropdownOption[]>;
  srosByDistrict: Map<string, DropdownOption[]>;
  villagesBySro: Map<string, DropdownOption[]>;
}

interface DropdownEntry {
  zone: { name: string; value: string };
  district: { name: string; value: string };
  subRegistrarOffice: { name: string; value: string };
  village: { name: string; value: string };
}

function processCascadingData(): CascadingDropdownData {
  const zones = new Map<string, DropdownOption>();
  const districtsByZone = new Map<string, Map<string, DropdownOption>>();
  const srosByDistrict = new Map<string, Map<string, DropdownOption>>();
  const villagesBySro = new Map<string, Map<string, DropdownOption>>();

  const dropdownData = dropdownDataRaw as DropdownEntry[];

  // Process each entry in the JSON
  dropdownData.forEach((entry) => {
    const zone = entry.zone;
    const district = entry.district;
    const sro = entry.subRegistrarOffice;
    const village = entry.village;

    // Add zone
    if (!zones.has(zone.value)) {
      zones.set(zone.value, { name: zone.name, value: zone.value });
    }

    // Add district to zone
    if (!districtsByZone.has(zone.value)) {
      districtsByZone.set(zone.value, new Map());
    }
    const districtsMap = districtsByZone.get(zone.value)!;
    if (!districtsMap.has(district.value)) {
      districtsMap.set(district.value, { name: district.name, value: district.value });
    }

    // Add SRO to district
    if (!srosByDistrict.has(district.value)) {
      srosByDistrict.set(district.value, new Map());
    }
    const srosMap = srosByDistrict.get(district.value)!;
    if (!srosMap.has(sro.value)) {
      srosMap.set(sro.value, { name: sro.name, value: sro.value });
    }

    // Add village to SRO
    if (!villagesBySro.has(sro.value)) {
      villagesBySro.set(sro.value, new Map());
    }
    const villagesMap = villagesBySro.get(sro.value)!;
    if (!villagesMap.has(village.value)) {
      villagesMap.set(village.value, { name: village.name, value: village.value });
    }
  });

  // Convert maps to arrays
  const result: CascadingDropdownData = {
    zones: Array.from(zones.values()).sort((a, b) => a.name.localeCompare(b.name)),
    districtsByZone: new Map(),
    srosByDistrict: new Map(),
    villagesBySro: new Map(),
  };

  districtsByZone.forEach((districtsMap, zoneValue) => {
    result.districtsByZone.set(
      zoneValue,
      Array.from(districtsMap.values()).sort((a, b) => a.name.localeCompare(b.name))
    );
  });

  srosByDistrict.forEach((srosMap, districtValue) => {
    result.srosByDistrict.set(
      districtValue,
      Array.from(srosMap.values()).sort((a, b) => a.name.localeCompare(b.name))
    );
  });

  villagesBySro.forEach((villagesMap, sroValue) => {
    result.villagesBySro.set(
      sroValue,
      Array.from(villagesMap.values()).sort((a, b) => a.name.localeCompare(b.name))
    );
  });

  return result;
}

// Build reverse lookup: village -> full hierarchy
export interface VillageHierarchy {
  village: DropdownOption;
  sro: DropdownOption;
  district: DropdownOption;
  zone: DropdownOption;
}

function buildVillageSearchMap(): Map<string, VillageHierarchy> {
  const villageMap = new Map<string, VillageHierarchy>();
  const dropdownData = dropdownDataRaw as DropdownEntry[];

  dropdownData.forEach((entry) => {
    const key = entry.village.value;
    if (!villageMap.has(key)) {
      villageMap.set(key, {
        village: { name: entry.village.name, value: entry.village.value },
        sro: { name: entry.subRegistrarOffice.name, value: entry.subRegistrarOffice.value },
        district: { name: entry.district.name, value: entry.district.value },
        zone: { name: entry.zone.name, value: entry.zone.value },
      });
    }
  });

  return villageMap;
}

export const cascadingData = processCascadingData();
export const villageSearchMap = buildVillageSearchMap();

// Get all unique villages for search
export function getAllVillages(): DropdownOption[] {
  const villages = Array.from(villageSearchMap.values()).map(h => h.village);
  const uniqueVillages = Array.from(
    new Map(villages.map(v => [v.value, v])).values()
  );
  return uniqueVillages.sort((a, b) => a.name.localeCompare(b.name));
}
