import { describe, it, expect, vi, beforeEach } from 'vitest';

const {
  mockInitializeApp,
  mockGetFirestore,
  mockGetStorage,
  mockGetAnalytics,
} = vi.hoisted(() => {
  const app = { name: '[DEFAULT]' };
  return {
    mockInitializeApp: vi.fn(() => app),
    mockGetFirestore: vi.fn(() => ({ __kind: 'firestore' })),
    mockGetStorage: vi.fn(() => ({ __kind: 'storage' })),
    mockGetAnalytics: vi.fn(() => ({ __kind: 'analytics' })),
  };
});

vi.mock('firebase/app', () => ({
  initializeApp: mockInitializeApp,
}));

vi.mock('firebase/firestore', () => ({
  getFirestore: mockGetFirestore,
}));

vi.mock('firebase/storage', () => ({
  getStorage: mockGetStorage,
}));

vi.mock('firebase/analytics', () => ({
  getAnalytics: mockGetAnalytics,
}));

describe('firebase bootstrap', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.unstubAllEnvs();
    mockInitializeApp.mockClear();
    mockGetFirestore.mockClear();
    mockGetStorage.mockClear();
    mockGetAnalytics.mockClear();
  });

  it('initializes the app and wires Firestore and Storage', async () => {
    const { app, db, storage, analytics } = await import('./firebase');

    expect(mockInitializeApp).toHaveBeenCalledTimes(1);
    expect(mockInitializeApp).toHaveBeenCalledWith(
      expect.objectContaining({
        projectId: 'capstonechexit',
        apiKey: expect.any(String),
      }),
    );

    const firebaseApp = mockInitializeApp.mock.results[0]?.value;
    expect(mockGetFirestore).toHaveBeenCalledWith(firebaseApp);
    expect(mockGetStorage).toHaveBeenCalledWith(firebaseApp);
    expect(mockGetAnalytics).not.toHaveBeenCalled();

    expect(db).toEqual({ __kind: 'firestore' });
    expect(storage).toEqual({ __kind: 'storage' });
    expect(analytics).toBeNull();
    expect(app).toBe(firebaseApp);
  });

  it('uses VITE_FIREBASE_PROJECT_ID from the environment when set', async () => {
    vi.stubEnv('VITE_FIREBASE_PROJECT_ID', 'from-env');
    await import('./firebase');

    expect(mockInitializeApp).toHaveBeenCalledWith(
      expect.objectContaining({ projectId: 'from-env' }),
    );
  });
});
