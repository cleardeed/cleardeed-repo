import { useState, useMemo, useEffect, useRef } from 'react';
import { ArrowLeft, Send } from 'lucide-react';
import {
  PropertyFormData,
  PropertyFormErrors,
} from '../types/property';
import { cascadingData } from '../utils/dropdownProcessor';

interface PropertyDetailsFormProps {
  onBack: () => void;
  onSubmit: (data: PropertyFormData) => void;
  initialData?: PropertyFormData;
  hideButtons?: boolean;
}

export const PropertyDetailsForm = ({
  onBack,
  onSubmit,
  initialData,
  hideButtons = false,
}: PropertyDetailsFormProps) => {
  const [formData, setFormData] = useState<PropertyFormData>(
    initialData || {
      zone: '',
      district: '',
      sro_name: '',
      village: '',
      survey_number: '',
      subdivision: '',
      email: '',
    }
  );

  const [errors, setErrors] = useState<PropertyFormErrors>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Update form data when initialData changes (e.g., from village search)
  useEffect(() => {
    if (initialData) {
      setFormData(initialData);
    }
  }, [initialData]);

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  const districtOptions = useMemo(() => {
    return formData.zone
      ? (cascadingData.districtsByZone.get(formData.zone) || [])
      : [];
  }, [formData.zone]);

  const sroOptions = useMemo(() => {
    return formData.district
      ? (cascadingData.srosByDistrict.get(formData.district) || [])
      : [];
  }, [formData.district]);

  const villageOptions = useMemo(() => {
    return formData.sro_name
      ? (cascadingData.villagesBySro.get(formData.sro_name) || [])
      : [];
  }, [formData.sro_name]);

  const validateField = (
    field: keyof PropertyFormData,
    value: string
  ): string | undefined => {
    switch (field) {
      case 'zone':
        return value ? undefined : 'Zone is required';
      case 'district':
        return value ? undefined : 'District is required';
      case 'sro_name':
        return value ? undefined : 'SRO Name is required';
      case 'village':
        return value ? undefined : 'Village is required';
      case 'survey_number':
        if (!value) return 'Survey Number is required';
        if (!/^[a-zA-Z0-9]+$/.test(value))
          return 'Survey Number must be alphanumeric only';
        if (value.length > 50) return 'Survey Number must be max 50 characters';
        return undefined;
      case 'subdivision':
        return undefined;
      case 'email':
        if (!value) return undefined; // Email is optional
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(value)) return 'Please enter a valid email address';
        return undefined;
      default:
        return undefined;
    }
  };

  const handleFieldChange = (
    field: keyof PropertyFormData,
    value: string
  ) => {
    setFormData((prev) => {
      const updated = { ...prev, [field]: value };

      if (field === 'zone') {
        updated.district = '';
        updated.sro_name = '';
        updated.village = '';
      } else if (field === 'district') {
        updated.sro_name = '';
        updated.village = '';
      } else if (field === 'sro_name') {
        updated.village = '';
      }

      // Update parent component with current data for inline form (debounced)
      if (hideButtons) {
        // Clear existing timer
        if (debounceTimerRef.current) {
          clearTimeout(debounceTimerRef.current);
        }
        // Set new timer - only submit after user stops typing for 500ms
        debounceTimerRef.current = setTimeout(() => {
          onSubmit(updated);
        }, 500);
      }

      return updated;
    });

    if (touched[field]) {
      const error = validateField(field, value);
      setErrors((prev) => {
        const updated = { ...prev };
        if (error) {
          updated[field] = error;
        } else {
          delete updated[field];
        }
        return updated;
      });
    }
  };

  const handleBlur = (field: keyof PropertyFormData) => {
    setTouched((prev) => ({ ...prev, [field]: true }));
    const error = validateField(field, formData[field]);
    setErrors((prev) => {
      const updated = { ...prev };
      if (error) {
        updated[field] = error;
      } else {
        delete updated[field];
      }
      return updated;
    });
  };

  const validateForm = (): boolean => {
    const newErrors: PropertyFormErrors = {};
    const requiredFields: Array<keyof PropertyFormData> = [
      'zone',
      'district',
      'sro_name',
      'village',
      'survey_number',
    ];

    requiredFields.forEach((field) => {
      const error = validateField(field, formData[field]);
      if (error) {
        newErrors[field] = error;
      }
    });

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    setTouched({
      zone: true,
      district: true,
      sro_name: true,
      village: true,
      survey_number: true,
      subdivision: true,
      email: true,
    });

    if (validateForm()) {
      onSubmit(formData);
    }
  };

  const FormField = ({
    label,
    required,
    error,
    children,
  }: {
    label: string;
    required: boolean;
    error?: string;
    children: React.ReactNode;
  }) => (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-2">
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
      </label>
      {children}
      {error && <p className="text-sm text-red-600 mt-1">{error}</p>}
    </div>
  );

  if (hideButtons) {
    return (
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* First Row - Zone and District */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <FormField
            label="Zone"
            required
            error={touched.zone ? errors.zone : undefined}
          >
            <select
              value={formData.zone}
              onChange={(e) => handleFieldChange('zone', e.target.value)}
              onBlur={() => handleBlur('zone')}
              className={`w-full px-3 py-2 text-sm border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition ${
                touched.zone && errors.zone
                  ? 'border-red-500 bg-red-50'
                  : 'border-gray-300'
              }`}
            >
              <option value="">Select Zone</option>
              {cascadingData.zones.map((zone) => (
                <option key={zone.value} value={zone.value}>
                  {zone.name}
                </option>
              ))}
            </select>
          </FormField>

          <FormField
            label="District"
            required
            error={touched.district ? errors.district : undefined}
          >
            <select
              value={formData.district}
              onChange={(e) => handleFieldChange('district', e.target.value)}
              onBlur={() => handleBlur('district')}
              disabled={!formData.zone}
              className={`w-full px-3 py-2 text-sm border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition ${
                !formData.zone
                  ? 'bg-gray-100 cursor-not-allowed opacity-60'
                  : ''
              } ${
                touched.district && errors.district
                  ? 'border-red-500 bg-red-50'
                  : 'border-gray-300'
              }`}
            >
              <option value="">
                {formData.zone ? 'Select District' : 'Select Zone first'}
              </option>
              {districtOptions.map((district) => (
                <option key={district.value} value={district.value}>
                  {district.name}
                </option>
              ))}
            </select>
          </FormField>
        </div>

        {/* Second Row - SRO and Village */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <FormField
            label="SRO Name"
            required
            error={touched.sro_name ? errors.sro_name : undefined}
          >
            <select
              value={formData.sro_name}
              onChange={(e) => handleFieldChange('sro_name', e.target.value)}
              onBlur={() => handleBlur('sro_name')}
              disabled={!formData.district}
              className={`w-full px-3 py-2 text-sm border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition ${
                !formData.district
                  ? 'bg-gray-100 cursor-not-allowed opacity-60'
                  : ''
              } ${
                touched.sro_name && errors.sro_name
                  ? 'border-red-500 bg-red-50'
                  : 'border-gray-300'
              }`}
            >
              <option value="">
                {formData.district ? 'Select SRO Name' : 'Select District first'}
              </option>
              {sroOptions.map((sro) => (
                <option key={sro.value} value={sro.value}>
                  {sro.name}
                </option>
              ))}
            </select>
          </FormField>

          <FormField
            label="Village"
            required
            error={touched.village ? errors.village : undefined}
          >
            <select
              value={formData.village}
              onChange={(e) => handleFieldChange('village', e.target.value)}
              onBlur={() => handleBlur('village')}
              disabled={!formData.sro_name}
              className={`w-full px-3 py-2 text-sm border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition ${
                !formData.sro_name
                  ? 'bg-gray-100 cursor-not-allowed opacity-60'
                  : ''
              } ${
                touched.village && errors.village
                  ? 'border-red-500 bg-red-50'
                  : 'border-gray-300'
              }`}
            >
              <option value="">
                {formData.sro_name ? 'Select Village' : 'Select SRO first'}
              </option>
              {villageOptions.map((village) => (
                <option key={village.value} value={village.value}>
                  {village.name}
                </option>
              ))}
            </select>
          </FormField>
        </div>

        {/* Third Row - Survey Number and Subdivision */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <FormField
            label="Survey Number"
            required
            error={touched.survey_number ? errors.survey_number : undefined}
          >
            <input
              type="text"
              value={formData.survey_number}
              onChange={(e) =>
                handleFieldChange('survey_number', e.target.value)
              }
              onBlur={() => handleBlur('survey_number')}
              placeholder="e.g., 12/1A, 45B2"
              className={`w-full px-3 py-2 text-sm border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition ${
                touched.survey_number && errors.survey_number
                  ? 'border-red-500 bg-red-50'
                  : 'border-gray-300'
              }`}
            />
          </FormField>

          <FormField
            label="Subdivision"
            required={false}
            error={touched.subdivision ? errors.subdivision : undefined}
          >
            <input
              type="text"
              value={formData.subdivision}
              onChange={(e) =>
                handleFieldChange('subdivision', e.target.value)
              }
              onBlur={() => handleBlur('subdivision')}
              placeholder="Optional"
              className={`w-full px-3 py-2 text-sm border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition border-gray-300`}
            />
          </FormField>
        </div>

        {/* Fourth Row - Email */}
        <div>
          <FormField
            label="Email ID"
            required={false}
            error={touched.email ? errors.email : undefined}
          >
            <input
              type="email"
              value={formData.email || ''}
              onChange={(e) =>
                handleFieldChange('email', e.target.value)
              }
              onBlur={() => handleBlur('email')}
              placeholder="your.email@example.com (Optional)"
              className={`w-full px-3 py-2 text-sm border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition ${
                touched.email && errors.email
                  ? 'border-red-500 bg-red-50'
                  : 'border-gray-300'
              }`}
            />
          </FormField>
        </div>
      </form>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="max-w-4xl mx-auto px-4 py-12">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Property Details
          </h1>
          <p className="text-gray-600">
            Enter the property information for validation
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <FormField
              label="Zone"
              required
              error={touched.zone ? errors.zone : undefined}
            >
              <select
                value={formData.zone}
                onChange={(e) => handleFieldChange('zone', e.target.value)}
                onBlur={() => handleBlur('zone')}
                className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition ${
                  touched.zone && errors.zone
                    ? 'border-red-500 bg-red-50'
                    : 'border-gray-300'
                }`}
              >
                <option value="">Select Zone</option>
                {cascadingData.zones.map((zone) => (
                  <option key={zone.value} value={zone.value}>
                    {zone.name}
                  </option>
                ))}
              </select>
            </FormField>

            <FormField
              label="District"
              required
              error={touched.district ? errors.district : undefined}
            >
              <select
                value={formData.district}
                onChange={(e) => handleFieldChange('district', e.target.value)}
                onBlur={() => handleBlur('district')}
                disabled={!formData.zone}
                className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition ${
                  !formData.zone
                    ? 'bg-gray-100 cursor-not-allowed opacity-60'
                    : ''
                } ${
                  touched.district && errors.district
                    ? 'border-red-500 bg-red-50'
                    : 'border-gray-300'
                }`}
              >
                <option value="">
                  {formData.zone ? 'Select District' : 'Select Zone first'}
                </option>
                {districtOptions.map((district) => (
                  <option key={district.value} value={district.value}>
                    {district.name}
                  </option>
                ))}
              </select>
            </FormField>

            <FormField
              label="SRO Name"
              required
              error={touched.sro_name ? errors.sro_name : undefined}
            >
              <select
                value={formData.sro_name}
                onChange={(e) => handleFieldChange('sro_name', e.target.value)}
                onBlur={() => handleBlur('sro_name')}
                disabled={!formData.district}
                className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition ${
                  !formData.district
                    ? 'bg-gray-100 cursor-not-allowed opacity-60'
                    : ''
                } ${
                  touched.sro_name && errors.sro_name
                    ? 'border-red-500 bg-red-50'
                    : 'border-gray-300'
                }`}
              >
                <option value="">
                  {formData.district ? 'Select SRO Name' : 'Select District first'}
                </option>
                {sroOptions.map((sro) => (
                  <option key={sro.value} value={sro.value}>
                    {sro.name}
                  </option>
                ))}
              </select>
            </FormField>

            <FormField
              label="Village"
              required
              error={touched.village ? errors.village : undefined}
            >
              <select
                value={formData.village}
                onChange={(e) => handleFieldChange('village', e.target.value)}
                onBlur={() => handleBlur('village')}
                disabled={!formData.sro_name}
                className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition ${
                  !formData.sro_name
                    ? 'bg-gray-100 cursor-not-allowed opacity-60'
                    : ''
                } ${
                  touched.village && errors.village
                    ? 'border-red-500 bg-red-50'
                    : 'border-gray-300'
                }`}
              >
                <option value="">
                  {formData.sro_name ? 'Select Village' : 'Select SRO first'}
                </option>
                {villageOptions.map((village) => (
                  <option key={village.value} value={village.value}>
                    {village.name}
                  </option>
                ))}
              </select>
            </FormField>

            <FormField
              label="Survey Number"
              required
              error={touched.survey_number ? errors.survey_number : undefined}
            >
              <input
                type="text"
                value={formData.survey_number}
                onChange={(e) =>
                  handleFieldChange('survey_number', e.target.value)
                }
                onBlur={() => handleBlur('survey_number')}
                placeholder="e.g., 12/1A, 45B2"
                className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition ${
                  touched.survey_number && errors.survey_number
                    ? 'border-red-500 bg-red-50'
                    : 'border-gray-300'
                }`}
              />
            </FormField>

            <FormField
              label="Subdivision"
              required={false}
              error={touched.subdivision ? errors.subdivision : undefined}
            >
              <input
                type="text"
                value={formData.subdivision}
                onChange={(e) =>
                  handleFieldChange('subdivision', e.target.value)
                }
                onBlur={() => handleBlur('subdivision')}
                placeholder="Optional subdivision information"
                className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition border-gray-300`}
              />
            </FormField>

            <FormField
              label="Email ID"
              required={false}
              error={touched.email ? errors.email : undefined}
            >
              <input
                type="email"
                value={formData.email || ''}
                onChange={(e) =>
                  handleFieldChange('email', e.target.value)
                }
                onBlur={() => handleBlur('email')}
                placeholder="your.email@example.com (Optional)"
                className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition ${
                  touched.email && errors.email
                    ? 'border-red-500 bg-red-50'
                    : 'border-gray-300'
                }`}
              />
            </FormField>

            {!hideButtons && (
              <div className="flex gap-4 pt-6">
                <button
                  type="button"
                  onClick={onBack}
                  className="flex items-center gap-2 px-6 py-3 rounded-lg font-semibold border-2 border-gray-300 text-gray-700 hover:bg-gray-50 transition-all duration-200"
                >
                  <ArrowLeft className="w-5 h-5" />
                  Back
                </button>

                <button
                  type="submit"
                  className="ml-auto flex items-center gap-2 px-8 py-3 rounded-lg font-semibold bg-blue-600 hover:bg-blue-700 text-white transition-all duration-200 shadow-md hover:shadow-lg"
                >
                  Submit
                  <Send className="w-5 h-5" />
                </button>
              </div>
            )}
          </form>
        </div>
      </div>
    </div>
  );
};