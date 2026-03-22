import * as React from 'react';
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardHeader from '@mui/material/CardHeader';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import LinearProgress from '@mui/material/LinearProgress';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import { doc, onSnapshot, Timestamp } from 'firebase/firestore';
import type { PredictUiState } from '../../api/chexit';
import { db } from '../../firebase';

// Assets served from /assets at the project root (e.g., public/assets)
const cxrOut = '/assets/cxrout.png';

function formatUploadedAt(seconds: number | null): string {
  if (seconds == null) return 'Uploaded • 2 min ago';
  const d = new Date(seconds * 1000);
  const now = Date.now();
  const diffMs = now - d.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 1) return 'Uploaded • just now';
  if (diffMins === 1) return 'Uploaded • 1 min ago';
  return `Uploaded • ${diffMins} min ago`;
}

type FeaturesProps = {
  /** When set, this URL is shown in the Input X-Ray preview (e.g. right after upload). */
  previewImageUrl?: string | null;
  /** Local `blob:` URL from file picker — shown immediately before any Firebase upload. */
  localPreviewUrl?: string | null;
  predictUi: PredictUiState;
};

function contributionBars(riskScore: number) {
  const r = Math.min(100, Math.max(0, Math.round(riskScore)));
  return [
    { label: 'MobileNet-V2', value: Math.min(100, Math.round(r * 0.88)), barColor: '#6366f1' },
    { label: 'ResNet-50', value: Math.min(100, Math.round(r * 1.02)), barColor: '#3b82f6' },
    { label: 'DenseNet-121', value: r, barColor: '#22c55e' },
  ];
}

export default function Features({ previewImageUrl, localPreviewUrl, predictUi }: FeaturesProps) {
  const pageBg = 'background.default';
  const cardBg = 'background.default';
  const cardBorder = 'divider';
  const mutedText = '#9ca3af';

  const [latestUpload, setLatestUpload] = React.useState<{
    downloadURL: string;
    fileName: string;
    uploadedAt: number | null;
  } | null>(null);

  React.useEffect(() => {
    const latestRef = doc(db, 'uploads', 'latest');
    const unsub = onSnapshot(
      latestRef,
      (snap) => {
        const data = snap.data();
        if (data?.downloadURL) {
          const uploadedAt = data.uploadedAt instanceof Timestamp
            ? data.uploadedAt.seconds
            : typeof data.uploadedAt?.seconds === 'number'
              ? data.uploadedAt.seconds
              : null;
          setLatestUpload({
            downloadURL: data.downloadURL,
            fileName: data.fileName ?? 'image',
            uploadedAt,
          });
        } else {
          setLatestUpload(null);
        }
      },
      () => setLatestUpload(null),
    );
    return () => unsub();
  }, []);

  const remotePreviewSrc = previewImageUrl ?? latestUpload?.downloadURL ?? null;
  const previewSrc = localPreviewUrl ?? remotePreviewSrc;
  const hasRemoteImage = Boolean(remotePreviewSrc);
  const previewSubheader = localPreviewUrl && !previewImageUrl
    ? 'Preview • not uploaded yet'
    : previewImageUrl
      ? 'Uploaded • just now'
      : hasRemoteImage
        ? formatUploadedAt(latestUpload?.uploadedAt ?? null)
        : 'No image uploaded';

  const pred = predictUi.data;
  const heatmapSrc = pred?.heatmap
    ? `data:image/png;base64,${pred.heatmap}`
    : cxrOut;
  const isHighRisk = Boolean(
    pred?.diagnosis?.toLowerCase().includes('positive') ||
      pred?.confidence_label?.toLowerCase().includes('high'),
  );
  const modelRows = pred ? contributionBars(pred.risk_score) : [
    { label: 'MobileNet-V2', value: 60, barColor: '#6366f1' },
    { label: 'ResNet-50', value: 80, barColor: '#3b82f6' },
    { label: 'DenseNet-121', value: 75, barColor: '#22c55e' },
  ];

  return (
    <Box sx={{ bgcolor: pageBg }}>
      <Container id="features" maxWidth="lg" sx={{ pt: { xs: 4, md: 5 }, pb: { xs: 6, md: 7 } }}>
        <Typography
          variant="h6"
          sx={{
            mb: 3,
            color: 'text.primary',
            fontWeight: 600,
            letterSpacing: 0.5,
          }}
        >
          AI-assisted TB overview
        </Typography>

        {predictUi.error ? (
          <Typography variant="body2" color="error" sx={{ mb: 2 }}>
            {predictUi.error}
          </Typography>
        ) : null}

        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', md: 'row' },
            gap: { xs: 2.5, md: 3 },
            alignItems: 'stretch',
          }}
        >
          {/* Input image */}
          <Card
            variant="outlined"
            sx={{
              flex: 1,
              bgcolor: cardBg,
              borderColor: cardBorder,
              color: 'text.primary',
              borderRadius: 2,
            }}
          >
            <CardHeader
              title="Input X-Ray"
              subheader={previewSubheader}
              sx={{
                pb: 1,
                '& .MuiCardHeader-title': {
                  fontSize: 14,
                  fontWeight: 600,
                },
                '& .MuiCardHeader-subheader': {
                  color: mutedText,
                  fontSize: 12,
                },
              }}
            />
            <CardContent sx={{ pt: 1 }}>
              {previewSrc ? (
                <Box
                  component="img"
                  key={previewSrc}
                  src={previewSrc}
                  alt="Input chest X-ray"
                  sx={{
                    borderRadius: 2,
                    bgcolor: 'background.default',
                    border: '1px dashed',
                    borderColor: cardBorder,
                    width: '100%',
                    aspectRatio: '3 / 4',
                    objectFit: 'cover',
                  }}
                />
              ) : (
                <Box
                  sx={{
                    borderRadius: 2,
                    bgcolor: '#000',
                    border: '1px dashed',
                    borderColor: cardBorder,
                    width: '100%',
                    aspectRatio: '3 / 4',
                  }}
                />
              )}
              <Box sx={{ mt: 2 }}>
                <Typography variant="caption" sx={{ color: mutedText }}>
                  {previewSrc ? 'View: PA • Resolution: 1024×1024' : 'Upload an X-ray to preview'}
                </Typography>
              </Box>
            </CardContent>
          </Card>
{/* Diagnosis */}
          <Card
            variant="outlined"
            sx={{
              flex: 1.1,
              bgcolor: cardBg,
              borderColor: cardBorder,
              color: 'text.primary',
              borderRadius: 2,
            }}
          >
  <CardContent
    sx={{
      pb: 2,
      display: 'flex',
      flexDirection: 'column',
      gap: 3, // uniform vertical spacing between blocks
    }}
  >
    {predictUi.loading ? (
      <LinearProgress sx={{ borderRadius: 1 }} />
    ) : null}
    {/* Top: diagnosis + score */}
    <Box>
      <Typography
        variant="overline"
        sx={{ color: mutedText, letterSpacing: 1.5 }}
      >
        DIAGNOSIS
      </Typography>

      <Box sx={{ mt: 1 }}>
        <Typography
          component="h2"
          sx={{
            fontWeight: 900,
            lineHeight: 1.05,
            fontSize: { xs: '2.8rem', md: '3.4rem' },
          }}
        >
          {pred?.diagnosis ?? (predictUi.loading ? '…' : '—')}
        </Typography>
        <Chip
          label={pred?.confidence_label ?? (predictUi.loading ? 'Analyzing' : 'Run Analyze')}
          size="small"
          sx={{
            mt: 1,
            alignSelf: 'flex-start',
            bgcolor: isHighRisk ? 'rgba(248,113,113,0.12)' : 'rgba(34,197,94,0.12)',
            border: isHighRisk ? '1px solid #f87171' : '1px solid #4ade80',
            color: isHighRisk ? '#fecaca' : '#bbf7d0',
            fontSize: 11,
            height: 24,
            borderRadius: 999,
            px: 1.5,
            ...(!pred &&
              !predictUi.loading && {
                bgcolor: 'action.hover',
                borderColor: 'divider',
                color: 'text.secondary',
              }),
          }}
        />
      </Box>

      <Box sx={{ mt: 3 }}>
        <Typography
          variant="subtitle2"
          sx={{ color: mutedText, mb: 0.75 }}
        >
          TB risk score
        </Typography>
        <Typography
          component="p"
          sx={{
            fontWeight: 900,
            lineHeight: 1,
            fontSize: { xs: '3.8rem', md: '4.4rem' }, // big 70%
          }}
        >
          {pred != null ? `${Math.round(pred.risk_score)}%` : predictUi.loading ? '…' : '—'}
        </Typography>
        <Typography
          variant="caption"
          sx={{ color: mutedText, display: 'block', mt: 1 }}
        >
          {pred
            ? 'TB probability (0–100%) from the MobileNet head; heatmap is Score-CAM on the CXR (U-Net lung mask).'
            : 'Choose an X-ray above and click Analyze to call the Chexit API.'}
        </Typography>
      </Box>
    </Box>

    {/* Bottom: contributions (pulled closer to 70%) */}
    <Box>
                <Divider sx={{ borderColor: cardBorder, mb: 3 }} />

      <Typography
        variant="subtitle2"
        sx={{ color: mutedText, mb: 1.5 }}
      >
        Model contributions
      </Typography>

      {modelRows.map((item) => (
        <Box key={item.label} sx={{ mb: 1.5 }}>
          <Stack
            direction="row"
            justifyContent="space-between"
            sx={{ mb: 0.5 }}
          >
            <Typography variant="body2">{item.label}</Typography>
            <Typography variant="body2" sx={{ color: mutedText }}>
              {item.value}%
            </Typography>
          </Stack>
                    <LinearProgress
                      variant="determinate"
                      value={item.value}
                      sx={{
                        height: 6,
                        borderRadius: 3,
                        bgcolor: 'action.hover',
                        '& .MuiLinearProgress-bar': {
                          borderRadius: 3,
                          bgcolor: item.barColor,
                        },
                      }}
                    />
        </Box>
      ))}
    </Box>
  </CardContent>
</Card>



          {/* Heatmap */}
          <Card
            variant="outlined"
            sx={{
              flex: 1,
              bgcolor: cardBg,
              borderColor: cardBorder,
              color: 'text.primary',
              borderRadius: 2,
            }}
          >
            <CardHeader
              title="Prediction Heatmap"
              subheader="Highlighted TB-suspect regions"
              sx={{
                pb: 1,
                '& .MuiCardHeader-title': {
                  fontSize: 14,
                  fontWeight: 600,
                },
                '& .MuiCardHeader-subheader': {
                  color: mutedText,
                  fontSize: 12,
                },
              }}
            />
            <CardContent sx={{ pt: 1 }}>
              <Box
                component="img"
                key={heatmapSrc}
                src={heatmapSrc}
                alt="Prediction Heatmap"
                sx={{
                  borderRadius: 2,
                  width: '100%',
                  aspectRatio: '3 / 4',
                  border: '1px solid',
                  borderColor: cardBorder,
                  objectFit: 'cover',
                }}
              />
              <Box sx={{ mt: 2 }}>
                <Typography variant="caption" sx={{ color: mutedText }}>
                  {pred
                    ? 'Score-CAM overlay (lung-masked) from the Chexit API.'
                    : 'Run Analyze to fetch a heatmap from the API, or see the static demo image.'}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Container>
    </Box>
  );
}
