import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { alprService } from '../../services/api';
import { toast } from 'react-hot-toast';
import {
  CheckIcon,
  XMarkIcon,
  MagnifyingGlassIcon,
} from '@heroicons/react/24/outline';

const VerificationQueue = () => {
  const [selectedLog, setSelectedLog] = useState(null);
  const [editedPlate, setEditedPlate] = useState('');
  const [editedProvince, setEditedProvince] = useState('');
  const [notes, setNotes] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const queryClient = useQueryClient();

  // Fetch pending verifications
  const { data, isLoading } = useQuery({
    queryKey: ['pending-verifications', currentPage],
    queryFn: () => alprService.getPendingVerifications(currentPage, 20),
  });

  // Verify mutation
  const verifyMutation = useMutation({
    mutationFn: ({ id, data }) => alprService.verifyDetection(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['pending-verifications']);
      toast.success('บันทึกสำเร็จ');
      setSelectedLog(null);
      resetForm();
    },
    onError: (error) => {
      toast.error('เกิดข้อผิดพลาด: ' + error.message);
    },
  });

  const logs = data?.data?.items || [];
  const totalPages = data?.data?.total_pages || 1;

  const handleSelectLog = (log) => {
    setSelectedLog(log);
    setEditedPlate(log.detected_plate);
    setEditedProvince(log.detected_province || '');
    setNotes('');
  };

  const handleConfirm = () => {
    if (!selectedLog) return;

    verifyMutation.mutate({
      id: selectedLog.id,
      data: {
        corrected_plate: editedPlate,
        corrected_province: editedProvince,
        status: 'MLPR',
        verified_by: 'operator', // TODO: Get from auth context
        verification_notes: notes,
      },
    });
  };

  const handleReject = () => {
    if (!selectedLog) return;

    verifyMutation.mutate({
      id: selectedLog.id,
      data: {
        corrected_plate: editedPlate,
        corrected_province: editedProvince,
        status: 'REJECTED',
        verified_by: 'operator',
        verification_notes: notes || 'ปฏิเสธการตรวจจับ',
      },
    });
  };

  const resetForm = () => {
    setEditedPlate('');
    setEditedProvince('');
    setNotes('');
  };

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">คิวตรวจสอบ</h1>
        <p className="text-gray-600 mt-1">
          ตรวจสอบและแก้ไขป้ายทะเบียนที่มีความเชื่อมั่นต่ำ
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* List */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">
            รอตรวจสอบ ({logs.length})
          </h2>

          {isLoading && (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
            </div>
          )}

          {!isLoading && logs.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <CheckIcon className="w-16 h-16 mx-auto mb-2 text-green-500" />
              <p>ไม่มีรายการรอตรวจสอบ</p>
            </div>
          )}

          <div className="space-y-3 max-h-[600px] overflow-y-auto">
            {logs.map((log) => (
              <div
                key={log.id}
                onClick={() => handleSelectLog(log)}
                className={`border rounded-lg p-4 cursor-pointer transition-all ${
                  selectedLog?.id === log.id
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-200 hover:border-primary-300'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <span className="text-2xl font-bold font-thai">
                        {log.detected_plate}
                      </span>
                      <span className={`badge ${
                        log.confidence_score > 0.8 
                          ? 'badge-warning' 
                          : 'badge-danger'
                      }`}>
                        {(log.confidence_score * 100).toFixed(0)}%
                      </span>
                    </div>
                    
                    {log.detected_province && (
                      <p className="text-sm text-gray-600 mt-1 font-thai">
                        {log.detected_province}
                      </p>
                    )}
                    
                    <p className="text-xs text-gray-500 mt-2">
                      {new Date(log.detection_timestamp).toLocaleString('th-TH')}
                    </p>
                  </div>

                  <div className="flex flex-col items-end space-y-2">
                    {log.plate_crop_path && (
                      <img
                        src={`/api/images/${log.plate_crop_path}`}
                        alt="Plate"
                        className="w-32 h-auto rounded border"
                      />
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex justify-center items-center space-x-2 mt-4">
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="btn btn-secondary"
              >
                ก่อนหน้า
              </button>
              <span className="text-sm text-gray-600">
                หน้า {currentPage} / {totalPages}
              </span>
              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="btn btn-secondary"
              >
                ถัดไป
              </button>
            </div>
          )}
        </div>

        {/* Editor */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">แก้ไขป้ายทะเบียน</h2>

          {!selectedLog && (
            <div className="text-center py-12 text-gray-500">
              <MagnifyingGlassIcon className="w-16 h-16 mx-auto mb-2" />
              <p>เลือกรายการที่ต้องการตรวจสอบ</p>
            </div>
          )}

          {selectedLog && (
            <div className="space-y-6">
              {/* Images */}
              <div className="space-y-4">
                {selectedLog.full_image_path && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      ภาพเต็ม
                    </label>
                    <img
                      src={`/api/images/${selectedLog.full_image_path}`}
                      alt="Full frame"
                      className="w-full rounded-lg border"
                    />
                  </div>
                )}

                {selectedLog.plate_crop_path && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      ป้ายทะเบียน (ซูม)
                    </label>
                    <img
                      src={`/api/images/${selectedLog.plate_crop_path}`}
                      alt="Plate crop"
                      className="w-full rounded-lg border"
                    />
                  </div>
                )}
              </div>

              {/* Detection Info */}
              <div className="bg-gray-50 rounded-lg p-4 space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">AI ตรวจจับ:</span>
                  <span className="font-bold font-thai">{selectedLog.detected_plate}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">จังหวัด:</span>
                  <span className="font-thai">{selectedLog.detected_province || '-'}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">ความเชื่อมั่น:</span>
                  <span>{(selectedLog.confidence_score * 100).toFixed(2)}%</span>
                </div>
              </div>

              {/* Edit Form */}
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    ป้ายทะเบียนที่ถูกต้อง *
                  </label>
                  <input
                    type="text"
                    value={editedPlate}
                    onChange={(e) => setEditedPlate(e.target.value)}
                    className="input font-thai text-lg"
                    placeholder="กก1234"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    จังหวัด *
                  </label>
                  <input
                    type="text"
                    value={editedProvince}
                    onChange={(e) => setEditedProvince(e.target.value)}
                    className="input font-thai"
                    placeholder="กรุงเทพมหานคร"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    หมายเหตุ (ถ้ามี)
                  </label>
                  <textarea
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    rows="3"
                    className="input font-thai"
                    placeholder="เพิ่มหมายเหตุ..."
                  />
                </div>
              </div>

              {/* Actions */}
              <div className="flex space-x-3">
                <button
                  onClick={handleConfirm}
                  disabled={!editedPlate || !editedProvince || verifyMutation.isPending}
                  className="btn btn-success flex-1 flex items-center justify-center space-x-2"
                >
                  <CheckIcon className="w-5 h-5" />
                  <span>ยืนยัน (MLPR)</span>
                </button>

                <button
                  onClick={handleReject}
                  disabled={verifyMutation.isPending}
                  className="btn btn-danger flex items-center justify-center space-x-2"
                >
                  <XMarkIcon className="w-5 h-5" />
                  <span>ปฏิเสธ</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VerificationQueue;