import { useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import {
  CameraView,
  useCameraPermissions,
  useMicrophonePermissions,
} from 'expo-camera';
import { StatusBar } from 'expo-status-bar';

import {
  AnalysisStatus,
  JobResult,
  uploadAndAnalyze,
} from '../src/api';

const STATUS_TEXT: Record<AnalysisStatus, string> = {
  idle: 'Ready',
  requesting_upload: 'Preparing secure upload…',
  uploading: 'Uploading video…',
  creating_job: 'Starting analysis…',
  analyzing: 'Tracking movement and analyzing technique…',
  done: 'Coaching report ready',
  error: 'Something went wrong',
};

export default function HomeScreen() {
  const cameraRef = useRef<CameraView | null>(null);
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [microphonePermission, requestMicrophonePermission] = useMicrophonePermissions();
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState<AnalysisStatus>('idle');
  const [result, setResult] = useState<JobResult | null>(null);

  const requestPermissions = async () => {
    const camera = cameraPermission?.granted
      ? cameraPermission
      : await requestCameraPermission();
    const microphone = microphonePermission?.granted
      ? microphonePermission
      : await requestMicrophonePermission();
    return camera.granted && microphone.granted;
  };

  const recordAndAnalyze = async () => {
    if (recording || status === 'analyzing' || status === 'uploading') return;

    const allowed = await requestPermissions();
    if (!allowed) {
      Alert.alert(
        'Camera permission needed',
        'Please allow camera and microphone access so The AI Athlete can record your video.',
      );
      return;
    }

    if (!cameraRef.current) return;

    try {
      setResult(null);
      setStatus('idle');
      setRecording(true);

      const captured = await cameraRef.current.recordAsync({
        maxDuration: 10,
      });
      setRecording(false);

      if (!captured?.uri) {
        throw new Error('No video was recorded.');
      }

      const report = await uploadAndAnalyze(captured.uri, {
        sport: 'tennis',
        focus: 'swing',
        onStatus: setStatus,
      });
      setResult(report);
    } catch (error) {
      setRecording(false);
      setStatus('error');
      Alert.alert(
        'Analysis failed',
        error instanceof Error ? error.message : 'Please try again.',
      );
    }
  };

  const stopRecording = () => {
    cameraRef.current?.stopRecording();
  };

  const busy =
    status === 'requesting_upload' ||
    status === 'uploading' ||
    status === 'creating_job' ||
    status === 'analyzing';

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="light" />
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.header}>
          <Text style={styles.brand}>THE AI ATHLETE</Text>
          <Text style={styles.title}>Your AI coach starts with one short video.</Text>
          <Text style={styles.subtitle}>
            V2 test mode: tennis swing analysis. Recording stops automatically at 10 seconds.
          </Text>
        </View>

        <View style={styles.cameraCard}>
          {cameraPermission?.granted ? (
            <CameraView
              ref={cameraRef}
              style={styles.camera}
              facing="back"
              mode="video"
              videoQuality="720p"
            />
          ) : (
            <View style={styles.permissionBox}>
              <Text style={styles.permissionText}>Camera access is required.</Text>
              <Pressable style={styles.secondaryButton} onPress={requestPermissions}>
                <Text style={styles.secondaryButtonText}>Enable Camera</Text>
              </Pressable>
            </View>
          )}
        </View>

        <View style={styles.actions}>
          {!recording ? (
            <Pressable
              style={[styles.primaryButton, busy && styles.disabled]}
              onPress={recordAndAnalyze}
              disabled={busy}
            >
              <Text style={styles.primaryButtonText}>
                {busy ? 'Working…' : 'Record 10s Tennis Swing'}
              </Text>
            </Pressable>
          ) : (
            <Pressable style={styles.stopButton} onPress={stopRecording}>
              <Text style={styles.primaryButtonText}>Stop & Analyze</Text>
            </Pressable>
          )}
        </View>

        <View style={styles.statusCard}>
          <View style={styles.statusRow}>
            {busy && <ActivityIndicator />}
            <Text style={styles.statusText}>
              {recording ? 'Recording…' : STATUS_TEXT[status]}
            </Text>
          </View>
          <Text style={styles.statusHint}>
            {status === 'uploading' && 'The video is uploading securely to temporary cloud storage.'}
            {status === 'analyzing' && 'Computer vision is measuring movement and preparing the coaching result.'}
            {status === 'done' && 'Review the priorities below, then record another attempt.'}
          </Text>
        </View>

        {result && (
          <View style={styles.reportCard}>
            <Text style={styles.reportTitle}>Coaching Report</Text>
            <Text style={styles.sport}>{result.sport?.toUpperCase() || 'SPORT'}</Text>
            {result.summary ? <Text style={styles.summary}>{result.summary}</Text> : null}

            {(result.analysis?.recommendations || []).slice(0, 3).map((tip, index) => (
              <View key={`${tip}-${index}`} style={styles.tipCard}>
                <Text style={styles.tipNumber}>{index + 1}</Text>
                <Text style={styles.tipText}>{tip}</Text>
              </View>
            ))}

            {(result.focus_tips || []).slice(0, 3).map((tip) => (
              <View key={`focus-${tip.id}`} style={styles.tipCard}>
                <Text style={styles.tipNumber}>•</Text>
                <Text style={styles.tipText}>{tip.text}</Text>
              </View>
            ))}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#0B0F14' },
  container: { padding: 20, paddingBottom: 48, gap: 18 },
  header: { gap: 8, marginTop: 8 },
  brand: { color: '#8AB4FF', fontSize: 13, fontWeight: '800', letterSpacing: 2 },
  title: { color: 'white', fontSize: 30, lineHeight: 36, fontWeight: '800' },
  subtitle: { color: '#A9B2BF', fontSize: 15, lineHeight: 21 },
  cameraCard: { borderRadius: 22, overflow: 'hidden', backgroundColor: '#171D25' },
  camera: { width: '100%', aspectRatio: 3 / 4 },
  permissionBox: { minHeight: 330, alignItems: 'center', justifyContent: 'center', gap: 16, padding: 24 },
  permissionText: { color: '#D8DEE8', fontSize: 16 },
  actions: { gap: 10 },
  primaryButton: { backgroundColor: '#2F6BFF', paddingVertical: 16, borderRadius: 16, alignItems: 'center' },
  stopButton: { backgroundColor: '#E5484D', paddingVertical: 16, borderRadius: 16, alignItems: 'center' },
  disabled: { opacity: 0.55 },
  primaryButtonText: { color: 'white', fontSize: 16, fontWeight: '800' },
  secondaryButton: { borderWidth: 1, borderColor: '#405064', paddingHorizontal: 20, paddingVertical: 12, borderRadius: 14 },
  secondaryButtonText: { color: 'white', fontWeight: '700' },
  statusCard: { backgroundColor: '#171D25', borderRadius: 18, padding: 16, gap: 8 },
  statusRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  statusText: { color: 'white', fontSize: 16, fontWeight: '700' },
  statusHint: { color: '#96A1AF', lineHeight: 19 },
  reportCard: { backgroundColor: 'white', borderRadius: 22, padding: 20, gap: 12 },
  reportTitle: { color: '#111827', fontSize: 23, fontWeight: '800' },
  sport: { color: '#2F6BFF', fontSize: 13, fontWeight: '800', letterSpacing: 1.5 },
  summary: { color: '#364152', fontSize: 16, lineHeight: 23 },
  tipCard: { flexDirection: 'row', gap: 12, backgroundColor: '#F3F6FA', borderRadius: 14, padding: 14 },
  tipNumber: { color: '#2F6BFF', fontSize: 17, fontWeight: '900' },
  tipText: { flex: 1, color: '#1F2937', fontSize: 15, lineHeight: 21 },
});
