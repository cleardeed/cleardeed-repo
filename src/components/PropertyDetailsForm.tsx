import { useState, useMemo } from 'react';
import { ArrowLeft, Send } from 'lucide-react';
import {
  PropertyFormData,
  PropertyFormErrors,
  DISTRICTS,
  SRO_BY_DISTRICT,
  VILLAGES_BY_SRO,
} from '../types/property';

interface PropertyDetailsFormProps {
  onBack: () => void;
  onSubmit: (data: PropertyFormData) => void;
}

export const PropertyDetailsForm = ({
  onBack,
  onSubmit,
}: PropertyDetailsFormProps) => {
  const [formData, setFormData] = useState<PropertyFormData>({
    district: '',
    sro_name: '',
    village: '',
    survey_number: '',
    subdivision: '',
  });

  const [errors, setErrors] = useState<PropertyFormErrors>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});

  const sroOptions = useMemo(() => {
    return formData.district
      ? (SRO_BY_DISTRICT[formData.district] || [])
      : [];
  }, [formData.district]);

  const villageOptions = useMemo(() => {
    if (!formData.sro_name || !formData.district) return [];
    return VILLAGES_BY_SRO[formData.sro_name] || [];
  }, [formData.sro_name, formData.district]);

  const validateField = (
    field: keyof PropertyFormData,
    value: string
  ): string | undefined => {
    switch (field) {
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

      if (field === 'district') {
        updated.sro_name = '';
        updated.village = '';
      } else if (field === 'sro_name') {
        updated.village = '';
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
      district: true,
      sro_name: true,
      village: true,
      survey_number: true,
      subdivision: true,
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
              label="District"
              required
              error={touched.district ? errors.district : undefined}
            >
              <select
                value={formData.district}
                onChange={(e) => handleFieldChange('district', e.target.value)}
                onBlur={() => handleBlur('district')}
                className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition ${
                  touched.district && errors.district
                    ? 'border-red-500 bg-red-50'
                    : 'border-gray-300'
                }`}
              >
                <option value="">Select District</option>
                {DISTRICTS.map((district) => (
                  <option key={district} value={district}>
                    {district}
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
                  <option key={sro} value={sro}>
                    {sro}
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
                  <option key={village} value={village}>
                    {village}
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
          </form>
        </div>
      </div>
    </div>
  );
};
